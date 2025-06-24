import sys 
import os
import random
import math

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import make_grid
import torchvision

from PIL import Image

import argparse

from collections import namedtuple
import torch
import torch.nn as nn

from quant import QTensor, quantize_tensor, dequantize_tensor

from collections import namedtuple



def get_model_size(model, bits):
    size_model = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            #size_model += param.numel() * torch.finfo(param.data.dtype).bits
            size_model += param.numel() * bits
        else:
            #size_model += param.numel() * torch.iinfo(param.data.dtype).bits
            size_model += param.numel() * bits
    #print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")
    return size_model


def matrix_dimensions(original_module_shape: torch.Size) -> Tuple[int, int]:
    shape = original_module_shape

    if len(shape) != 4 and len(shape) != 2:
        assert "Invalid shape"

    if len(shape) == 4:  # conv2d layer
        n_dim = shape[0]
        m_dim = shape[1] * shape[2] * shape[3]
    else:  # linear layer
        n_dim = shape[0]
        m_dim = shape[1]

    return n_dim, m_dim

def get_residual_lora(dw, rank, delta_A=torch.tensor(0.), delta_B=torch.tensor(0.)):

    r"""
    mu = dw.mean()
    std = dw.std()
    
    dw_ = dw.clamp(mu - 0.5*std, mu + 0.5*std)
    dw = dw - dw_.detach()
    """

    rows, cols = matrix_dimensions(dw.shape)
    dw = dw.reshape(rows, cols)

    if cols > rows:
       v_mat, s_mat, u_mat = dw.t().svd()
    else:
       u_mat, s_mat, v_mat = dw.svd()

    if rank > s_mat.shape[0]:
        #print("rank: ", rank)
        #print("s_mat.shape: ", s_mat.shape)
        rank=s_mat.shape[0]

    s_mat = s_mat.narrow(0, 0, rank)
    u_mat = u_mat.narrow(1, 0, rank)
    v_mat = v_mat.narrow(1, 0, rank)

    #print("s_mat.shape: ", s_mat.shape)

    s_mat_sqrt = torch.sqrt(s_mat)

    a_mat = s_mat_sqrt * v_mat
    b_mat = s_mat_sqrt * u_mat

    return rank, a_mat, b_mat

def _quantize_weights(module, num_bits, k):
    residual_weights = dict()
    for n, m in module.named_modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            w=m.weight.clone().detach()
            with torch.no_grad():
                w_q = quantize_tensor(w, num_bits=num_bits, k=k)
                w_q = dequantize_tensor(w_q)
                m.weight.copy_(w_q.detach())
                residual_weights[n] = w - w_q
    return residual_weights

def _quantize_lora_weights(lora_dict, lora_num_bits):

    for n in lora_dict:
        m = lora_dict[n]
        w = m.weight.clone().detach()
        with torch.no_grad():
            w_q = quantize_tensor(w, num_bits=lora_num_bits, k=None)
            w_q = dequantize_tensor(w_q)
            m.weight.copy_(w_q.detach())




class CoRaModel(nn.Module):
    def __init__(   self, 
                    model, 
                    num_bits=4,         #quantization bits for main.
                    lora_num_bits=None, #quantization bits for lora.
                    k=3.0):

        super().__init__()
 
        self.main = model
        self.num_bits = num_bits
        self.lora_num_bits = lora_num_bits
        self.k = k

        #quantize the weights in main branch.
        self.quantize_weights()

        #lora branch.
        self.lora = dict()

        #hooks.
        self.input_quant_hooks = []
        self.lora_hooks = []

        #calibration stats for pre-forward.
        #self.pre_forward_stats = dict()

        #sometimes we need to disable/enable hooks..we store the information here.
        self.hook_config = dict()


        #quantize inputs before forward.
        def input_quant_hook_fn(module_, inputs_):

            n=module_.name

            #pass-through.
            if not self.hook_config[n]["input_quant"]["enable"]:
                return inputs_

            inputs_=inputs_[0]

            k = self.k
            
            r"""
            if k is not None:
                mu = inputs_.mean()
                std = inputs_.std()
                alpha = mu - k * std
                beta = mu + k * std
                inputs_ = quantize_tensor(inputs_, num_bits=self.num_bits, alpha=alpha, beta=beta, k=k)
            else:
                inputs_ = quantize_tensor(inputs_, num_bits=self.num_bits, k=None)
            """

            inputs_ = quantize_tensor(inputs_, num_bits=self.num_bits, k=None)
            inputs_ = dequantize_tensor(inputs_)

            return (inputs_)

        #implementing lora.
        def lora_hook_fn(module_, inputs_, outputs_):
            n=module_.name
            #pass-through.
            if not self.hook_config[n]["lora"]["enable"]:
                return outputs_

            inputs_ = inputs_[0]


            group_reshape = False

            if isinstance(module_, torch.nn.Conv2d):
                batch_size, chans, w, h = inputs_.shape
                groups = module_.groups
                in_channels = module_.in_channels
                #handling groups.
                if groups == in_channels and groups > 1:
                    group_reshape = True
            
            if group_reshape:
                inputs_ = inputs_.reshape(batch_size*chans, 1, w, h)

            #compute lora branch.
            B=self.lora[n+".B"]
            A=self.lora[n+".A"]
            z = A(inputs_)

            z = B(z)

            if group_reshape:
                _, rank, w_, h_ = z.shape
                z = z.reshape(batch_size, rank, w_, h_)
                print("z.shape: ", z.shape)

            outputs_ = outputs_ + z
                
            return outputs_
 
      
        """
            register hooks.
        """
        for n, m in self.main.named_modules():
            m.name = n

            #if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            if isinstance(m, torch.nn.Conv2d):
                lora_hook = m.register_forward_hook(lora_hook_fn)
                self.lora_hooks.append(lora_hook)

                input_quant_hook = m.register_forward_pre_hook(input_quant_hook_fn)
                self.input_quant_hooks.append(input_quant_hook)

                #we disable the hook by default at this moment.
                self.hook_config[n] = dict()

                self.hook_config[n]["lora"] = dict()
                self.hook_config[n]["lora"]["enable"] = False

                self.hook_config[n]["input_quant"] = dict()
                self.hook_config[n]["input_quant"]["enable"] = False



    #quantize main branch.
    def quantize_weights(self):
        self.residual_weights = _quantize_weights( module=self.main, 
                                                    num_bits = self.num_bits, 
                                                    k = self.k)

    #quantize lora branch to e.g. 16bits.
    #experiments show min-max clipping outperforms others.
    def quantize_lora_weights(self):
        _quantize_lora_weights(lora_dict = self.lora, 
                                lora_num_bits = self.lora_num_bits)

          
    def get_lora_rank_hard(self, norm_budget):
        ranks_dict = dict()

        min_rank=1
        for n, m in self.main.named_modules():
            if isinstance(m, torch.nn.Conv2d):

                in_channels=m.in_channels
                out_channels=m.out_channels

                #hard-thresholding ranks.
                rank=min(int(in_channels*norm_budget), int(out_channels*norm_budget))
                rank=max(min_rank, int(rank))

                ranks_dict[n] = rank
            
            r"""
            if isinstance(m, torch.nn.Linear):

                in_features=m.in_features
                out_features=m.out_features

                rank=min(int(in_features*norm_budget), int(out_features*norm_budget))
                rank=max(min_rank, int(rank))

                ranks_dict[n] = rank
            """

        return ranks_dict

    def init_lora(self, ranks_dict, delta_weights_dict=None):

        self.lora = dict()


        """
            setup lora.
        """
        for n, m in self.main.named_modules():
            m.name = n

            if isinstance(m, torch.nn.Conv2d):
                #e.g. Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

                in_channels=m.in_channels
                out_channels=m.out_channels
                kernel_size=m.kernel_size
                stride=m.stride
                padding=m.padding
                bias=m.bias
                groups=m.groups

                #if groups > 1:
                #    continue

                #assert groups < 2

                #print(m)
                #print(m.weight.shape)
                #exit()


                r"""
                    transferring residual knowledge.

                """
                rank = ranks_dict[n]
                dw = self.residual_weights[n]

                if delta_weights_dict:
                    delta_A = torch.tensor(delta_weights_dict[n + ".A"])
                    delta_B = torch.tensor(delta_weights_dict[n + ".B"])
                else:
                    delta_A = torch.tensor(0.)
                    delta_B = torch.tensor(0.)
                    

                rank, a_mat, b_mat = get_residual_lora(dw, rank, delta_A, delta_B)
                ranks_dict[n] = rank

                #handling groups.
                in_channels = in_channels // groups

                self.lora[n + ".A"] = torch.nn.Conv2d(in_channels=in_channels,
                                                    out_channels=rank,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    bias=False,
                                                    padding=padding,
                                                    groups=1)

                self.lora[n + ".B"] = torch.nn.Conv2d(in_channels=rank,
                                                    out_channels=out_channels,
                                                    kernel_size=(1, 1), 
                                                    bias=False)

                with torch.no_grad():
                    
                    if groups > 1:
                        print()
                        print("groups: ", groups)
                        print("dw.shape: ", dw.shape)
                        print("a_mat.shape: ", a_mat.t().shape)
                        print(self.lora[n + ".A"])
                        print(self.lora[n + ".A"].weight.shape)
                        print("in_channels: ", in_channels)
                        print("rank: ", rank)
                        print("kernel_size: ", kernel_size)


                    a_mat = a_mat.t().reshape(rank, in_channels, kernel_size[0], kernel_size[1])
                    b_mat = b_mat.reshape(out_channels, rank, 1, 1)

                    self.lora[n + ".A"].weight.copy_(a_mat)
                    self.lora[n + ".B"].weight.copy_(b_mat)

            r"""

            if isinstance(m, torch.nn.Linear):
                continue

                in_features=m.in_features
                out_features=m.out_features


                #transferring residual knowledge.
                rank = ranks_dict[n]
                dw = self.residual_weights[n]
                rank, a_mat, b_mat = get_residual_lora(dw, rank)
                ranks_dict[n] = rank

                self.lora[n + ".A"] = torch.nn.Linear(in_features=in_features,
                                                    out_features=rank)

                self.lora[n + ".B"] = torch.nn.Linear(in_features=rank,
                                                    out_features=out_features)
                with torch.no_grad():
                    self.lora[n + ".A"].weight.copy_(a_mat.t())
                    self.lora[n + ".B"].weight.copy_(b_mat)
            """

    
        self.lora = nn.ParameterDict(self.lora)

        if self.lora_num_bits:
            self.quantize_lora_weights()

        self.print_size_stat()

    def print_size_stat(self):
        fp_model_size = get_model_size(self.main, 4)
        quant_model_size = get_model_size(self.main, self.num_bits/8.0)

        if self.lora_num_bits:
            lora_num_bits = self.lora_num_bits
        else:
            lora_num_bits = 32

        lora_size = get_model_size(self.lora, lora_num_bits/8.0)
        lora_ratio = (lora_size/fp_model_size)*100.

        #the ratio of the lora parameters to overall
        budget = get_model_size(self.lora, 1) / get_model_size(self.main, 1)

        

        #lora_ratio = budget * 100.
        #lora_size = budget * fp_model_size * (lora_num_bits/32.)

        equiv_bit_width=self.num_bits + lora_num_bits * budget

        print()
        print("CoRa:")
        print(f"num_bits: {self.num_bits}")
        print(f"lora_num_bits: {self.lora_num_bits}")
        print(f"k: {self.k}")
        print(f"budget: {budget:.2f}")
        print(f"model_size (FP):   {fp_model_size/1048576.0:.2f} MB")
        print(f"model_size (QINT): {quant_model_size/1048576.0:.2f} MB")
        print(f"lora_size:         {lora_size/1048576.0:.2f} MB")
        print(f"lora_ratio:        {lora_ratio:.2f}%")
        print(f"equiv_bit_width:   {equiv_bit_width:.2f}")
        print()
    

    def __enable_disable_hook_config(self, key: str, enable: bool):
        for n in self.hook_config:
            self.hook_config[n][key]["enable"] = enable

    def disable_lora(self):
        self.__enable_disable_hook_config("lora", False)

    def enable_lora(self):
        self.__enable_disable_hook_config("lora", True)


    def disable_input_quant(self):
        self.__enable_disable_hook_config("input_quant", False)

    def enable_input_quant(self):
        self.__enable_disable_hook_config("input_quant", True)

    #collecting activation dataset.
    def calibrate(self, dataset):
        raise NotImplementedError()

    def __del__(self):
        for hook in self.input_quant_hooks:
            hook.remove()

        for hook in self.lora_hooks:
            hook.remove()

    def forward(self, x):
        return self.main(x)


def test(args):

    resnet18 = torchvision.models.resnet18(pretrained=True)
    
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)

    args = parser.parse_args()

    print(args)
    test(args)

