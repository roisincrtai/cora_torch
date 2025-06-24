import sys 
import os
import random
import math

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

#QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])

class QTensor:
    def __init__(self, tensor, scale, zero_point, mu=0.0, std=1.0):
        self.tensor = tensor
        self.scale = scale
        self.zero_point = zero_point
        self.mu = mu
        self.std = std

def compute_zero_point(alpha, beta, num_bits):
    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale = (beta - alpha) / (qmax - qmin)

    zero_point = qmin - alpha / scale

    zero_point = zero_point.clamp(qmin, qmax)

    zero_point = int(zero_point)

    return scale, zero_point


r"""
    The [alpha, beta] is determined by 
"""
def quantize_tensor(x, num_bits, alpha=None, beta=None, k=2.0):

    if not alpha and not beta:
        if k is None:
            alpha = x.min()
            beta = x.max()
        else:
            mu = x.mean()
            std = x.std()
            alpha = mu - k * std
            beta = mu + k * std

        #print(f"max: {x.min()} max {x.max()} mu {mu} std {std} alpha {alpha} beta {beta}")

    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale, zero_point = compute_zero_point(alpha, beta, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()

    q_x = q_x.round().long()


    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)

def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)

r"""
    straight-through estimator.

"""
class FakeQuantizeSTEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits, k):
        x_q = quantize_tensor(x, num_bits, k)
        x_q = dequantize_tensor(x_q)
        return x_q

    #STE implementation.
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output



def test(args):

    resnet18 = torchvision.models.resnet18(pretrained=True)

    print(resnet18)

    bits_list = [8,7,6,5,4,3,2,1]
    errors_list_1 = []
    errors_list_2 = []

    for b in bits_list:
        w = resnet18.layer2[0].conv1.weight
        #w = resnet18.conv1.weight

        w_q = quantize_tensor(w, num_bits=b)
        w_ = dequantize_tensor(w_q)
        error = torch.mean(torch.abs(w - w_)).detach().numpy()
        errors_list_1.append(error)

        w_q = quantize_tensor(w, num_bits=b, k = None)
        w_ = dequantize_tensor(w_q)
        error = torch.mean(torch.abs(w - w_)).detach().numpy()
        errors_list_2.append(error)

        

    x_data=list(reversed(bits_list))

    plt.plot(x_data, errors_list_1, label=r"$[-k\sigma+\mu, k\sigma+\mu]$")
    plt.plot(x_data, errors_list_2, label=r"$min-max$")
    plt.xticks(x_data, bits_list)
    plt.ylabel("quantization error")
    plt.xlabel("bits")
    plt.xlim([1, 8])
    plt.legend()
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)

    args = parser.parse_args()

    print(args)
    test(args)

