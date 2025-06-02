import torch
import numpy as np

def MSE(target,model_output):

    xhat = model_output[0]

    return ((target - xhat)**2).sum(dim=-1).mean(),0

def sparse_MSE(target,model_output):

    
    z = model_output[1]
    return MSE(target,model_output), z.abs().sum(dim=-1).mean()

def ELBO(target,model_output,recon_precision=1):


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
    cross_entropy = d*np.log(2*np.pi)/2 + torch.einsum('bk,bk->b',z,z)/2
    
    return neg_lp.mean(), (-entropy + cross_entropy).mean()

def ELBO_more_stable(target,model_output,recon_precision=1):

    (xhat,z,dist) = model_output
    B,d = xhat.shape
    (mu,L,D) = dist
    L = L.squeeze() # goes from B x d x 1 -> B x d

    err = target - xhat
    neg_lp = torch.einsum('bd,bd->b',err,err) *(recon_precision)/2 + \
        d*np.log(2*np.pi)/2 - d * np.log(recon_precision)/2 

    t12 = -1/2 *torch.log(D).sum(dim=-1) - 1/2*torch.log((1 + torch.einsum('bd,bd->b',L/D,L)))#torch.log(torch.prod(D,dim=-1)*(1 + torch.einsum('bd,bd->b',L/D,L)))
    t22 = 1/2 * (D.sum(dim=-1) + (L**2).sum(dim=-1))
    t32 = - d/2
    t42 = 1/2 * (mu**2).sum(dim=-1)

    kl = (t12 + t22 + t32 + t42)

    return neg_lp.mean(),kl.mean()

def ELBO_linear_encouragement(target,model_output,model_decoder,recon_precision=1,weight_penalty=1):


    (xhat,z,dist) = model_output
    B,d = xhat.shape
    (mu,L,D) = dist
    L = L.squeeze() # goes from B x d x 1 -> B x d

    err = target - xhat
    neg_lp = torch.einsum('bd,bd->b',err,err) *(recon_precision)/2 + \
        d*np.log(2*np.pi)/2 - d * np.log(recon_precision)/2 

    t12 = -1/2 *torch.log(D).sum(dim=-1) - 1/2*torch.log((1 + torch.einsum('bd,bd->b',L/D,L)))#torch.log(torch.prod(D,dim=-1)*(1 + torch.einsum('bd,bd->b',L/D,L)))
    t22 = 1/2 * (D.sum(dim=-1) + (L**2).sum(dim=-1))
    t32 = - d/2
    t42 = 1/2 * (mu**2).sum(dim=-1)

    kl = (t12 + t22 + t32 + t42)

    penalty = torch.tensor([0.],requires_grad=True,device=model_decoder.device)
    for name,w in model_decoder.named_parameters():

        if 'nonlinearity' in name:
            penalty = penalty + (w**2).sum()

    return neg_lp.mean(),kl.mean()+penalty * weight_penalty

