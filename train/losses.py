import torch

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
    neg_lp = torch.linalg.norm(target - xhat,dim=-1,ord=2)**2 *(recon_precision)/2 + \
        torch.log(2*torch.pi)/2 - d * torch.log(recon_precision)/2 # doublecheck this
    
    entropy = dist.entropy()
    cross_entropy = 0.

    return neg_lp, entropy + cross_entropy