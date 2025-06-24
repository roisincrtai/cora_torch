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

from classifier_trainer import evaluate

from cora import get_model_size, matrix_dimensions, _quantize_weights, get_residual_lora

from differentiable_butterworth_filter import get_differentiable_butterworth_kernel

from cora_loss import CoRaLoss

from metric import Metric

r"""
    Quantizing lora weights with STE (straight-through estimator).
    See paper: Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation

"""
class QuantizedLoRaSTEGradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, num_bits, k=None):
        w_q = quantize_tensor(w, num_bits = num_bits, k=k)
        w_q = dequantize_tensor(w_q)
        return w_q

    #STE implementation.
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


r"""
    Differentiable solver for obtaining optimal ranks.

"""

class CoRaSolver(nn.Module):
    def __init__(   self, 
                    model,              #fp model.
                    num_bits=4,         #quantization bits for main.
                    lora_num_bits=None, #quantization bits for lora.
                    k=3.0,
                    init_norm_optimal_rank=0.3,   #norm cutoff rank. e.g. 0.3 = 30% of the ranks.
                    init_order = 6.0,
                    ):

        super().__init__()
 
        self.main = model
        self.num_bits = num_bits
        self.lora_num_bits = lora_num_bits
        self.k = k

        self.init_norm_optimal_rank = init_norm_optimal_rank
        self.init_order = init_order

        #quantize the weights in main branch.
        self.quantize_weights()

        #hooks.
        self.solver_hooks = []
        self.solver_data = dict()
        self.solver_norm_optimal_ranks = dict()

        #lora branch.
        self.lora = dict()
        self.rank_weights = dict()

        self.init_solver(init_norm_optimal_rank = init_norm_optimal_rank)
        self.init_full_rank_lora()


        def differentiable_rank_thresholding_fn(module_, inputs_, outputs_):

            n=module_.name

            if not isinstance(module_, nn.Conv2d):
                raise NotImplementedError()


            u_mat = self.solver_data[n]["u_mat"]
            s_mat_sqrt = self.solver_data[n]["s_mat_sqrt"]
            v_mat = self.solver_data[n]["v_mat"]
            unnorm_full_rank = self.solver_data[n]["unnorm_full_rank"]
            norm_cutoff_rank = self.solver_norm_optimal_ranks[n]


            #butterworth kernel.
            t_mat = get_differentiable_butterworth_kernel(
                                                fc_norm = norm_cutoff_rank,
                                                max_freq_unnorm = unnorm_full_rank,
                                                order = self.init_order)

            #send to gpu if possible.
            t_mat = t_mat.to(s_mat_sqrt.device)


            r"""
                differentiable DSP/Butterworth low-pass filtering.

            """
            #gradients not flow to U, S, V matrices.

            r"""
                !!! differentiable thresholding with Butterworth kernel.
                so we transform the non-differentiable problem into a differentiable problem.
                and we can use standard gradient descending to solve the optimal ranks.

                we allow gradient propagation on Butterworth kernel, but stop the gradients at S matrix.

            """

            s_mat_sqrt = t_mat * s_mat_sqrt.detach()


            a_mat = s_mat_sqrt * v_mat.detach()
            b_mat = s_mat_sqrt * u_mat.detach()

            rank = unnorm_full_rank

            r"""
            if isinstance(module_, nn.Conv2d):
                self.lora[n + ".A"].weight.data = a_mat.t().reshape(rank, module_.in_channels, module_.kernel_size[0], module_.kernel_size[1])

                self.lora[n + ".B"].weight.data = b_mat.reshape(module_.out_channels, rank, 1, 1)


            if isinstance(module_, nn.Linear):
                raise NotImplementedError()
                #self.lora[n + ".A"].weight.data = a_mat.t()
                #self.lora[n + ".B"].weight.data = b_mat
            """

            #compute lora branch.

            A = a_mat.t().reshape(rank, module_.in_channels, module_.kernel_size[0], module_.kernel_size[1])
            B = b_mat.reshape(module_.out_channels, rank, 1, 1)

            A_ref = self.lora[n + ".A"]
            B_ref = self.lora[n + ".B"]

            r"""
                weights are differentiable as well.

            """
            
            #this enables the changed weight differentiable 
            z = torch.func.functional_call(self.lora[n + ".A"], {"weight": A}, inputs_[0])
            z = torch.func.functional_call(self.lora[n + ".B"], {"weight": B}, z)

            r"""
            z = F.conv2d(inputs_[0], A, A_ref.bias, A_ref.stride,
                        A_ref.padding, A_ref.dilation, A_ref.groups)

            z = F.conv2d(z, B, B_ref.bias, B_ref.stride,
                        B_ref.padding, B_ref.dilation, B_ref.groups)
            """

            #B=self.lora[n+".B"]
            #A=self.lora[n+".A"]
            #z = A(inputs_[0])
            #z = B(z)

            outputs_ = outputs_ + z

            return outputs_



        """
            register hooks.
        """
        for n, m in self.main.named_modules():
            m.name = n

            #if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            if isinstance(m, torch.nn.Conv2d):

                solver_hook = m.register_forward_hook(differentiable_rank_thresholding_fn)
                self.solver_hooks.append(solver_hook)


    def init_solver(self, init_norm_optimal_rank):

        r"""
            we equip each layer a cutoff rank (rc).
        """
        main_weights_total=0.0
        for n, m in self.main.named_modules():
            #if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            if isinstance(m, torch.nn.Conv2d):

                self.solver_norm_optimal_ranks[n] = nn.Parameter(torch.tensor(init_norm_optimal_rank))

                dw = self.residual_weights[n]
                
                rows, cols = matrix_dimensions(dw.shape)
                dw = dw.reshape(rows, cols)

                if cols > rows:
                    v_mat, s_mat, u_mat = dw.t().svd()
                else:
                    u_mat, s_mat, v_mat = dw.svd()

                s_mat_sqrt = torch.sqrt(s_mat)

                v_mat.requires_grad=False
                u_mat.requires_grad=False
                s_mat_sqrt.requires_grad=False

                self.solver_data[n] = dict()
                self.solver_data[n]["u_mat"] = u_mat 
                self.solver_data[n]["v_mat"] = v_mat
                self.solver_data[n]["s_mat_sqrt"] = s_mat_sqrt
                self.solver_data[n]["unnorm_full_rank"] = len(s_mat_sqrt)

                #num_params=0.0
                for param in m.parameters():
                    #num_params += param.numel()
                    main_weights_total += param.numel()

                lora_weights=(v_mat.shape[0] + u_mat.shape[0]) * s_mat.shape[0]

                #self.rank_weights[n] = num_params
                self.rank_weights[n] = lora_weights

        #normalize rank_weights.
        #rank_weights_total = np.sum(list(self.rank_weights.values()))
        for n in self.rank_weights:
            self.rank_weights[n] = self.rank_weights[n] / main_weights_total

        #rank weights.
        print()
        print("rank weights:")
        for n in self.rank_weights:
            print(f"{n:<24} --> {self.rank_weights[n]:.6f}")

    def init_full_rank_lora(self):

        self.lora = dict()


        """
            setup lora.
        """
        for n, m in self.main.named_modules():
            m.name = n

            if isinstance(m, torch.nn.Linear):
                pass
                m.weight.requires_grad=False
                #if hasattr(m, 'bias'):
                #    m.bias.requires_grad=False

            if isinstance(m, torch.nn.Conv2d):
                #e.g. Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

                m.weight.requires_grad=False
                #if hasattr(m, 'bias'):
                #    m.bias.requires_grad=False

                in_channels=m.in_channels
                out_channels=m.out_channels
                kernel_size=m.kernel_size
                stride=m.stride
                padding=m.padding
                bias=m.bias
                groups=m.groups

                if groups > 1:
                    continue

                rank = self.solver_data[n]["unnorm_full_rank"]

                self.lora[n + ".A"] = torch.nn.Conv2d(in_channels=in_channels,
                                                    out_channels=rank,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    bias=False,
                                                    padding=padding)

                self.lora[n + ".B"] = torch.nn.Conv2d(in_channels=rank,
                                                    out_channels=out_channels,
                                                    kernel_size=(1, 1), 
                                                    bias=False)


                r"""
                    transferring residual knowledge.

                """
                dw = self.residual_weights[n]
                new_rank, a_mat, b_mat = get_residual_lora(dw, rank)

                with torch.no_grad():
                    self.lora[n + ".A"].weight.copy_(a_mat.t().reshape(rank, in_channels, kernel_size[0], kernel_size[1]))
                    self.lora[n + ".B"].weight.copy_(b_mat.reshape(out_channels, rank, 1, 1))


                self.lora[n + ".A"].weight.requires_grad=False
                self.lora[n + ".B"].weight.requires_grad=False

            if isinstance(m, torch.nn.Linear):
                continue

                in_features=m.in_features
                out_features=m.out_features

                rank = self.solver_data[n]["unnorm_full_rank"]

                self.lora[n + ".A"] = torch.nn.Linear(in_features=in_features,
                                                    out_features=rank)

                self.lora[n + ".B"] = torch.nn.Linear(in_features=rank,
                                                    out_features=out_features)

                r"""
                    transferring residual knowledge.

                """

                dw = self.residual_weights[n]
                new_rank, a_mat, b_mat = get_residual_lora(dw, rank)

                with torch.no_grad():
                    self.lora[n + ".A"].weight.copy_(a_mat.t())
                    self.lora[n + ".B"].weight.copy_(b_mat)


    
        self.lora = nn.ParameterDict(self.lora)

        #if self.lora_num_bits:
        #    self.quantize_lora_weights()


    #quantize main branch.
    def quantize_weights(self):

        self.residual_weights = _quantize_weights( module=self.main, 
                                                    num_bits = self.num_bits, 
                                                    k = self.k)

    def load_state(self, solution_file):
        pass

    def __del__(self):
        for hook in self.solver_hooks:
            hook.remove()

    def forward(self, x):
        return self.main(x)


def solve_optimal_ranks(solver, 
                        dataloader_opt, 
                        dataloader_val, 
                        teacher, 
                        optimizer, 
                        epochs, 
                        loss_temp, 
                        loss_gamma, 
                        loss_norm_target_budget,
                        loss_exp=True,
                        per_epoch_callback=None,
                        per_epoch_callback_data=None,
                        device=torch.device("cpu")):

    solver.train()
    solver.main.train()

    criterion = CoRaLoss(temp=loss_temp, gamma = loss_gamma)

    delta_weights_dict = dict()

    print()
    for n in solver.lora:
        print("dump weight: ", n)
        delta_weights_dict[n] = solver.lora[n].weight.detach().clone().cpu().numpy()


    #we use gradient descending.

    metric = Metric()

    keys = solver.solver_norm_optimal_ranks.keys()
    keys = list(sorted(keys))

    init_norm_ranks = [solver.solver_norm_optimal_ranks[k] for k in keys]
    init_norm_ranks = nn.utils.parameters_to_vector(init_norm_ranks).tolist()

    def get_ranks_dict(solver):
        ranks_dict = dict()
        for n in solver.solver_norm_optimal_ranks:
            norm_rank = solver.solver_norm_optimal_ranks[n].detach().cpu().item()
            if n in solver.solver_data:
                full_rank = solver.solver_data[n]["unnorm_full_rank"]
                ranks_dict[n] = np.ceil(full_rank * norm_rank).astype('int')
                if ranks_dict[n] < 1:
                    ranks_dict[n] = 1

        return ranks_dict

    init_ranks_dict = get_ranks_dict(solver)

    def get_min_norm_ranks_dict(solver):
        min_norm_optimal_ranks=dict()
        for n in solver.solver_data:
            min_norm_optimal_ranks[n] = 1.0 / solver.solver_data[n]["unnorm_full_rank"]
        return min_norm_optimal_ranks

    min_norm_optimal_ranks = get_min_norm_ranks_dict(solver)

    print("")
    print("min_norm_optimal_ranks: ", min_norm_optimal_ranks)

    def clamp_solution(solver):
        for n in solver.solver_norm_optimal_ranks:
            min_norm_optimal_rank = min_norm_optimal_ranks[n]

            if torch.isnan(solver.solver_norm_optimal_ranks[n]):
                with torch.no_grad():
                    solver.solver_norm_optimal_ranks[n].copy_(min_norm_optimal_rank)
            else:
                with torch.no_grad():
                    solver.solver_norm_optimal_ranks[n].clamp_(min_norm_optimal_rank, 1)


    def get_solution(init_ranks_dict, optimal_ranks_dict, metric):
        #ranks_dict = get_ranks_dict(solver)

        #print()
        #print("Solution:")
        #for n in ranks_dict:
        #    print(f"{n} --> {ranks_dict[n]}")
        #print()

        #print()
        #for n in delta_weights_dict:
        #    print("delta weight: ", n)
        #    w_old = delta_weights_dict[n]
        #    w_new = solver.lora[n].weight.detach().clone().cpu().numpy()
        #    delta_weights_dict[n] = w_new - w_old

        soldata = dict()
        soldata["init_ranks_dict"] = init_ranks_dict
        soldata["ranks_dict"] = optimal_ranks_dict
        soldata["metric_dict"] = metric.metric_dict
        #soldata["delta_weights_dict"] = delta_weights_dict
        
        return soldata



    metric = Metric()

    prev_norm_ranks = np.ones(len(keys))

    for epoch in range(1, epochs + 1):
        for batch_idx, (x, y) in enumerate(dataloader_opt, 1):
            x = x.to(device)

            #with torch.no_grad():
            #    teacher_logits = teacher(x)

            teacher_logits = torch.nn.functional.one_hot(y, num_classes=1000)
            teacher_logits = teacher_logits.float().to(device)

            #print(teacher_logits.shape)
            #exit()

            quant_logits = solver(x)



            norm_ranks = [solver.solver_norm_optimal_ranks[k] for k in keys]
            norm_ranks = nn.utils.parameters_to_vector(norm_ranks)

            rank_weights = [solver.rank_weights[k] for k in keys]
            rank_weights = torch.tensor(rank_weights).float()

            prev_norm_ranks = norm_ranks

            norm_ranks = norm_ranks.to(device)
            rank_weights = rank_weights.to(device)

            

            batch_loss, batch_kld_loss, batch_running_norm_budget = criterion(quant_logits = quant_logits, 
                            teacher_logits = teacher_logits, 
                            norm_ranks = norm_ranks,
                            rank_weights = rank_weights,
                            norm_target_budget = torch.tensor(loss_norm_target_budget),
                            exp = loss_exp)

            #optimizing.
            optimizer.zero_grad()
            batch_loss.backward()

            #gradient clipping
            for n in solver.solver_norm_optimal_ranks:
                torch.nn.utils.clip_grad_value_(solver.solver_norm_optimal_ranks[n], clip_value=0.2)

            optimizer.step()

            clamp_solution(solver)

            metric["kld"] = batch_kld_loss.detach().cpu().item()
            metric["running_norm_budget"] = batch_running_norm_budget.detach().cpu().item()
            metric["loss"] = batch_loss.detach().cpu().item()


            #computing accuracy
            _, y_hat = torch.max(quant_logits.detach().cpu().data, dim = 1)
            batch_accuracy = (y.detach().data == y_hat).float().mean()

            
            r"""
                evaluate
            """
            
            #no need to track gradients
            r"""
            for batch_idx_val, (x_val, y_val) in enumerate(dataloader_val, 1):
                with torch.no_grad():
                    x_val, y_val = x_val.to(device), y_val.to(device)
                
                    y_val_hat = solver(x_val)
        
                    if hasattr(y_val_hat, "logits"):
                        y_val_hat = y_val_hat.logits

                    _, y_val_hat_ = torch.max(y_val_hat.detach().data, dim = 1)
                    val_accuracy = (y_val.detach().data == y_val_hat_).float().mean()
                    val_accuracy = val_accuracy.detach().cpu().item()
            """

            metric["loss"] = batch_loss.detach().cpu().item()
            metric["accuracy"] = batch_accuracy.detach().cpu().item()
            #metric["val_accuracy"] = val_accuracy

            norm_ranks = [solver.solver_norm_optimal_ranks[k] for k in keys]
            norm_ranks = nn.utils.parameters_to_vector(norm_ranks).tolist()
            norm_ranks = np.array(norm_ranks)

            solution_error = np.abs(np.sum(norm_ranks - prev_norm_ranks.detach().cpu().numpy()))

            metric["solution_error"] = solution_error

            full_ranks = {n:solver.solver_data[n]["unnorm_full_rank"] for n in solver.solver_data}
            full_ranks = [full_ranks[k] for k in keys]
            full_ranks = np.array(full_ranks)

            lora_size = np.sum(norm_ranks * rank_weights.detach().cpu().numpy())

            metric["lora_size"] = lora_size


            num_bits = solver.num_bits
            r"""
            lora_num_bits = solver.lora_num_bits
            if lora_num_bits is None:
                lora_num_bits = 32
            """

            lora_num_bits=8
            equiv_bit_width = num_bits + lora_num_bits * lora_size

            end_str='\n'
            print('Optimizing {} [{}/{} ({:.0f}%)] -- loss batch {:.6f} epoch {:.6f} -- batch acc {:.2f}% avg acc {:.2f}% -- running budget {:.6f} -- equiv_bit_width {:.2f} (m=8) -- solution_error {:.4f}'.format(
                    epoch, batch_idx, 
                    len(dataloader_opt), 
                    100. * (batch_idx / len(dataloader_opt)),

                    batch_loss, 
                    metric["loss"],
                    
                    100. * batch_accuracy,
                    100. * metric["accuracy"],
                    
                    lora_size,
                    equiv_bit_width,

                    solution_error
                    ), end = end_str)

            optimal_ranks = np.ceil(full_ranks * norm_ranks).astype('int')
            print("optimal unnorm ranks: ", optimal_ranks)
            norm_ranks_=[f"{r:.3f}" for r in norm_ranks]
            print("optimal norm ranks:   ", norm_ranks_)
            init_norm_ranks_ = [f"{r:.3f}" for r in init_norm_ranks]
            #print("init norm ranks:      ", init_norm_ranks_)

            r"""
            evaluate(solver, 
                     device, 
                     dataloader,
                     criterion = torch.nn.CrossEntropyLoss(),
                     running_batchnorm = True,
                     max_batches = 10)
            """

        if per_epoch_callback is not None and callable(per_epoch_callback):

            optimal_ranks_dict = get_ranks_dict(solver)
            soldata = get_solution(init_ranks_dict, optimal_ranks_dict, metric)

            per_epoch_callback(solver, epoch, soldata, per_epoch_callback_data)
 
    r"""
    ranks_dict = get_ranks_dict(solver)

    print()
    print("Solution:")
    for n in ranks_dict:
        print(f"{n} --> {ranks_dict[n]}")
    print()

    print()
    for n in delta_weights_dict:
        print("delta weight: ", n)
        w_old = delta_weights_dict[n]
        w_new = solver.lora[n].weight.detach().clone().cpu().numpy()
        delta_weights_dict[n] = w_new - w_old

    soldata = dict()
    soldata["init_ranks_dict"] = init_ranks_dict
    soldata["ranks_dict"] = ranks_dict
    soldata["metric_dict"] = metric.metric_dict
    #soldata["delta_weights_dict"] = delta_weights_dict
    """

    optimal_ranks_dict = get_ranks_dict(solver)
    soldata = get_solution(init_ranks_dict, optimal_ranks_dict, metric)

    return soldata


def test(args):
    #resnet18 = torchvision.models.resnet18(pretrained=True)
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)

    args = parser.parse_args()

    print(args)
    test(args)

