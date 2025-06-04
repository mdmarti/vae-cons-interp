import torch
import numpy as np


def linear_encouragement_prelu(model,weight=1):

    penalty = torch.Tensor([0.]).to(model.device)
    for name,w in model.named_parameters():

        
        if 'nonlinearity' in name:
            penalty = penalty + ((w - 1).abs()).pow(2).sum()

    return penalty * weight

def linear_encouragement_mat(model,weight=1):

    penalty = torch.Tensor([0.]).to(model.device)
    for name,w in model.named_parameters():

        if 'nonlinearity' in name:
            penalty = penalty + w.abs().pow(2).sum()

    return penalty * weight

