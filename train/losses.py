import torch
import numpy as np

def MSE(target,model_output):

    xhat = model_output[0]

    return ((target - xhat)**2).sum(dim=-1).mean(),0

def sparse_MSE(target,model_output):

    
    z = model_output[1]
    return MSE(target,model_output), z.abs().sum(dim=-1).mean()

def ELBO(target,model_output,recon_precision=1e-5):


    assert len(model_output) ==3, \
        print(f"model output must contain xhat, z, and latent distribution: instead has {len(model_output)} elements")

    (xhat,z,dist) = model_output

    B,d = xhat.shape
    err = target - xhat
    neg_lp = torch.einsum('bd,bd->b',err,err) *(recon_precision)/2 + \
        d*np.log(2*np.pi)/2 - d * np.log(recon_precision)/2 # doublecheck this

    ### for the kl term, this is maybe not the most stable -- let's return the parameters instead of 
    ### the distribution
    entropy = dist.entropy()
    cross_entropy = d*np.log(2*np.pi)/2 + torch.einsum('bk,bk ->b',z,z)
    
    return neg_lp.mean(), (-entropy + cross_entropy).mean()