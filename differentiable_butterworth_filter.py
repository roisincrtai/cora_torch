import sys 
import os
import random
import math

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse



r"""
    Return a differentiable butterworth filter kernel.

"""

def get_differentiable_butterworth_kernel(  fc_norm: torch.tensor, 
                                            max_freq_unnorm: int, 
                                            order: torch.tensor) -> torch.tensor:


    #numeric stability.
    fc_norm_ = torch.nan_to_num(fc_norm, nan=1e-4, posinf=1.0)
    with torch.no_grad():
        fc_norm.copy_(fc_norm_)

    T = []

    for freq_unnorm_i in range(max_freq_unnorm):
        freq_norm_i = freq_unnorm_i / max_freq_unnorm

        f = freq_norm_i / fc_norm
        f = f.clamp(0, 1e6)

        T_i = torch.sqrt(1.0 + torch.pow(f, 2.0*order))
        #T_i = torch.sqrt(1.0 + (freq_norm_i / fc_norm)**(2.0*order))

        assert not torch.isnan(T_i).any()

        T.append(1.0 / T_i)

    T = torch.stack(T) #.to(fc.device)
    T = T.reshape(max_freq_unnorm).contiguous()

    T = T / T.max().detach()
    T.clamp(0, 1)

    return T


def show_kernel(args):
    fc_norm=torch.tensor(0.1)
    order=torch.tensor(8.0)
    max_freq_unnorm=60 #i.e. the max index of singular values

    T = get_differentiable_butterworth_kernel(fc_norm, max_freq_unnorm, order)
    
    print(T.shape)
    plt.plot(T.detach().numpy(), color="red", linestyle="solid", label="singular value weight")
    plt.axvline(x=(fc_norm*max_freq_unnorm), label="cutoff", linestyle="dotted")
    plt.xlim([0, max_freq_unnorm - 1])
    plt.ylim([0, 1.2])
    plt.ylabel("weight")
    plt.xlabel("rank")
    plt.grid("on")
    plt.legend()
    plt.title("Differentiable Butterworth kernel")
    plt.show()


def test(args):
    show_kernel(args)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)

    args = parser.parse_args()

    print(args)
    test(args)

