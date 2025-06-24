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

from cora import CoRaModel
from cora_solver import CoRaSolver, solve_optimal_ranks
from cora_loss import CoRaLoss

from classifier_trainer import evaluate, Metric
from load_save_model import load_model_state, save_model_state, load_model, save_model

from find_gpu import find_gpu

from dataset_helpers import get_torchvision_dataset, split_dataset, sample_subset, DatasetLabelNoise
from misc import seeds_everything

from pretrained_classifiers import get_torchvision_model

import copy

import pickle

def main(args):

    device = find_gpu()

    ds_mean = 0.5 
    ds_std = 0.5 
    
    seeds_everything(args.random_seed)

    dataset = get_torchvision_dataset(name = args.dataset,
                                      image_size = args.image_size,
                                      normalize_channel = True,
                                      ds_mean = ds_mean,
                                      ds_std = ds_std)

    dataset_train, dataset_val = split_dataset(dataset=dataset, num_classes=args.num_classes, split = args.dataset_split)

    x,y = next(iter(dataset_val))
    print("dataset x.shape: ", x.shape)

    dataloader_opt = torch.utils.data.DataLoader(dataset_train,
                                         batch_size = args.batch_size,
                                         shuffle = True)

    x_data=[]
    y_data=[]
    for batch_idx, (x, y) in enumerate(dataloader_opt, 1):
        x_data.append(x)
        y_data.append(y)
        if batch_idx > 5:
            break

    x_data = torch.cat(x_data, dim=0)
    y_data = torch.cat(y_data, dim=0)

    print("x_data.shape: ", x_data.shape)
    print("y_data.shape: ", y_data.shape)

    dataset_val = torch.utils.data.TensorDataset(x_data, y_data)

    dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                         batch_size = args.batch_size,
                                         shuffle = True)

    if "torchvision/" in args.model_savepath:
        name = args.model_savepath.split("/")[1]
        model = get_torchvision_model(name, pretrained = True)
    elif "dino/" in args.model_savepath:
        name = args.model_savepath.split("/")[1]
        model = get_torchvision_model("dino_" + name, pretrained = True)
    else:
        model = torch.load(args.model_savepath)



    criterion = torch.nn.CrossEntropyLoss()

    model.to(device)

  


    r"""
    evaluate(model, 
             device, 
             dataloader,
             criterion,
             max_batches = 10)
    """


    r"""
        solver.

    """

    teacher = copy.deepcopy(model)

    if args.init_norm_optimal_rank is None:
        init_norm_optimal_rank = args.loss_norm_target_budget
    else:
        init_norm_optimal_rank = args.init_norm_optimal_rank

    solver = CoRaSolver(model, 
                        num_bits=args.num_bits, 
                        lora_num_bits=args.lora_num_bits, 
                        k=args.k, 
                        init_norm_optimal_rank=init_norm_optimal_rank,
                        init_order=args.init_order)



    #we would like to optimize the cutoff ranks.

    parameters = list(solver.solver_norm_optimal_ranks.values())

    if args.optimize_lora == 1:
        parameters += list(solver.lora.parameters())

    from lamb_optimizer import Lamb

    r"""
    solver_optimizer = Lamb(parameters, 
                                        lr = args.learning_rate, 
                                        weight_decay=1e-5)
    """

    #solver_optimizer = torch.optim.Adam(parameters, lr = args.learning_rate, weight_decay=1e-5)
    solver_optimizer = torch.optim.Adam(parameters, lr = args.learning_rate)
    

    loss_exp = True
    if args.loss_exp == 0:
        loss_exp = False

    print("solution_file: ", args.solution_file)


    def per_epoch_callback(solver, epoch, soldata, callback_data):

        result_dir = os.path.normpath(os.path.dirname(args.solution_file)).rstrip(os.path.sep)

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        print("saving: ", result_dir)
        soldata["args_dict"] = vars(args)
        print(soldata)
        pickle.dump(soldata , open(args.solution_file, 'wb'))


    soldata = solve_optimal_ranks(solver = solver, 
                                     dataloader_opt = dataloader_opt,
                                     dataloader_val = dataloader_val,
                                     teacher = teacher, 
                                     optimizer = solver_optimizer, 
                                     epochs = args.epochs,
                                     loss_temp=args.loss_temp, 
                                     loss_gamma = args.loss_gamma,
                                     loss_norm_target_budget = args.loss_norm_target_budget,
                                     loss_exp = loss_exp,
                                     
                                     per_epoch_callback=per_epoch_callback,
                                     per_epoch_callback_data=(args, ),

                                     device = device)




 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #e.g. --model_savepath=torchvision/resnet18 or --model_savepath=save_models/model.pth
    parser.add_argument("--model_savepath", type=str, default=None)

    parser.add_argument("--num_bits", type=int, default=4, choices=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

    parser.add_argument("--k", type=float, default=3.0) #3.0

    parser.add_argument("--init_norm_optimal_rank", type=float, default=0.9)
    parser.add_argument("--init_order", type=float, default=4.0)

    parser.add_argument("--optimize_lora", type=int, default=0, choices=[0,1])

    parser.add_argument("--loss_exp", type=int, default=1, choices=[0,1])
    parser.add_argument("--loss_temp", type=float, default=1.0)
    parser.add_argument("--loss_gamma", type=float, default=1.0)
    parser.add_argument("--loss_norm_target_budget", type=float, default=None, required=True)

    parser.add_argument("--solution_file", type=str, default="results/solution_dict.pickle", required=True)

    parser.add_argument("--learning_rate", type=float, default=0.01)

    parser.add_argument("--finetune_lora", type=int, default=0, choices=[0,1])

    parser.add_argument("--disable_input_quant", type=int, default=1, choices=[0,1])
    parser.add_argument("--disable_lora", type=int, default=0, choices=[0,1])

    parser.add_argument("--lora_num_bits", type=int, default=None, choices=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)

    #parser.add_argument("--dataset", type=str, default="imagenet2012_val")
    #parser.add_argument("--dataset_split", type=float, default=0.9)

    parser.add_argument("--dataset", type=str, default="imagenet_mini")
    parser.add_argument("--dataset_split", type=float, default=0.82)

    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--image_size", type=int, default=224)

    #parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--random_seed", type=int, default=199)

    args = parser.parse_args()

    #print(vars(args))
    #exit()

    print(args)
    main(args)

