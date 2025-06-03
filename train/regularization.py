import torch
import numpy as np


def linear_encouragement_prelu(decoder,weight=1):

    penalty = torch.Tensor([0.]).to(decoder.device)
    for name,w in decoder.named_parameters():

        
        if 'nonlinearity' in name:
            penalty = penalty + ((w - 1).abs()).pow(2).sum()

    return penalty * weight

def linear_encouragement_mat(decoder,weight=1):

    penalty = torch.Tensor([0.]).to(decoder.device)
    for name,w in decoder.named_parameters():

        if 'nonlinearity' in name:
            penalty = penalty + w.abs().pow(2).sum()

    return penalty * weight

