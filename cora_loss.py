
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse

from typing import Union

class CoRaLoss(nn.Module):
    def __init__(self, temp, gamma):
        super().__init__()
 
        self.gamma = gamma
        self.temp = temp

    def forward(self, 
                quant_logits: torch.tensor, 
                teacher_logits: torch.tensor,
                norm_ranks: torch.tensor,
                rank_weights: Union[torch.tensor, np.array],
                norm_target_budget: Union[torch.tensor, np.array],
                exp = True)->torch.tensor:

        quant_prob = torch.softmax(quant_logits / self.temp, dim=1)

        #teacher_prob = nn.functional.softmax(teacher_logits / self.temp, dim=1)
        teacher_prob = teacher_logits

        #stop gradients on teacher.
        teacher_prob = teacher_prob.detach()

        r"""
            KLD

        """
        eps=1e-6
        kld_loss=-torch.sum(teacher_prob*torch.log(quant_prob + eps))
        kld_loss = kld_loss / quant_logits.shape[0]

        r"""
            budget loss. See the paper.

        """

        #running budgets.
        EPS=1e-4
        norm_ranks.clamp(EPS, 1.0)

        running_norm_budget=torch.sum(norm_ranks * rank_weights)
        budget_loss = torch.nn.functional.relu(running_norm_budget - norm_target_budget)

        #if running_norm_budget > norm_target_budget:
        #    budget_loss = running_norm_budget - norm_target_budget
        #else:
        #    budget_loss = (running_norm_budget - norm_target_budget)**2

        #if r is above target. give gradients. otherwise zero gradients.
        if exp == True:
            budget_loss = torch.exp(budget_loss)
        
        budget_loss = budget_loss * self.gamma

        loss = kld_loss + budget_loss

        print(f"kld: {kld_loss.detach().cpu().item(): .6f}")
        print(f"running_norm_budget: {running_norm_budget.detach().cpu().item(): .6f}")

        return loss, kld_loss, running_norm_budget


def test(args):
    criterion = CoRaLoss()

    quant_logits = torch.rand(10, 1000)
    teacher_logits = torch.rand(10, 1000)

    norm_ranks = torch.rand(70)
    weights = torch.rand(70)

    weights = weights / torch.sum(weights)

    loss = criterion(quant_logits, teacher_logits, norm_ranks, weights)

    print(loss)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)

    args = parser.parse_args()

    print(args)
    test(args)

