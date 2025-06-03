import torch
import numpy as np


def linear_encouragement_prelu(decoder,weight=1):

    penalty = torch.Tensor([0.],device=decoder.device,requires_grad=True)
    for name,w in decoder.named_parameters():

        
        if 'nonlinearity' in name:
            penalty = penalty + ((w - 1).abs()).sum()

    return penalty * weight

def linear_encouragement_mat(decoder,weight=1):

    penalty = torch.Tensor([0.],device=decoder.device,requires_grad=True)
    for name,w in decoder.named_parameters():

        if 'nonlinearity' in name:
            penalty = penalty + w.abs().sum()

    return penalty * weight

