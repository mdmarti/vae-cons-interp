import torch.nn as nn
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
import numpy as np
from data.toy_data import *
from data.data_utils import *
from train.losses import *
from train.train import train

from models.vae import *
import os
from fire import fire

from sklearn.mixture import GaussianMixture as GMM

def run_experiments(save_dir,n_samples=15000,proj_dim = 1000,linear=True,identity=False):


    if identity:

        proj = IdentityProjection(data_dim=2)

    elif linear:

        proj = LinearProjection(data_dim=2,project_dim=proj_dim)

    else:

        nonlinearity = lambda x: np.log(x + 1) if x > 0 else -np.log(-x + 1)
        inverse_nonlinearity = lambda x: np.exp(x) - 1 if x > 0 else -np.exp(-x) + 1 

        proj = NonlinearProjection(data_dim=2,project_dim=proj_dim,\
                                   nonlinearity=nonlinearity,inverse_nonlinearity=inverse_nonlinearity)


    latents,data,labels = generate_mixture_dataset(n_samples=n_samples,projection=proj,proj_sd=1.5)

    l1_lip_proj = l1norm(proj.w)

    base_model = GMM(n_components=4,covariance_type='full',n_init=10)
    pred_labels = base_model.fit_predict(data)

    loaders = get_loaders(data,test_size=0.4,seed=777,num_workers = os.cpu_count()//2,batch_size=512)

    precisions = np.logspace(0.01,10,10)
    loss = lambda target, model_out: ELBO(target,model_out,recon_precision=precisions[0])
    ### simple linear model #####
    enc_linearmodel = ProbabilisticEncoder(n_layers_shared=4,n_layers_private=3,data_dim=proj_dim,hidden_dim=125,latent_dim=2)
    dec_linearmodel = Decoder(n_layers=0,data_dim=proj_dim,hidden_dim=0,latent_dim=2,activation=nn.Identity())

    vae_linearmodel = VariationalAutoEncoder(enc_linearmodel,dec_linearmodel,LowRankMultivariateNormal)

    vae_linearmodel,opt_linearmodel,scheduler_linearmodel,reconstructions_linearmodel,regs_linearmodel=train(vae_linearmodel,loaders,loss=None)
    #############################

    #### deep linear model ######
    enc_deeplinearmodel = ProbabilisticEncoder(n_layers_shared=4,n_layers_private=3,data_dim=proj_dim,hidden_dim=125,latent_dim=2)
    dec_deeplinearmodel = Decoder(n_layers=7,data_dim=proj_dim,hidden_dim=125,latent_dim=2,activation=nn.Identity())

    vae_linearmodel = VariationalAutoEncoder(enc_deeplinearmodel,dec_deeplinearmodel,LowRankMultivariateNormal)
    ##############################


    #### nonlinear model #########
    enc_nonlinearmodel = ProbabilisticEncoder(n_layers_shared=4,n_layers_private=3,data_dim=proj_dim,hidden_dim=125,latent_dim=2)
    dec_nonlinearmodel = Decoder(n_layers=7,data_dim=proj_dim,hidden_dim=125,latent_dim=2)

    vae_nonlinearmodel = VariationalAutoEncoder(enc_nonlinearmodel,dec_nonlinearmodel,LowRankMultivariateNormal)
    ##############################

    #### lipschitz model #########
    enc_lipschitz = ProbabilisticEncoder(n_layers_shared=4,n_layers_private=3,data_dim=proj_dim,hidden_dim=125,latent_dim=2)
    dec_lipschitz = LipschitzDecoder(n_layers=0,data_dim=proj_dim,hidden_dim=125,latent_dim=2,max_lipschitz=l1_lip_proj,norm_func=l1norm)

    vae_nonlinearmodel = VariationalAutoEncoder(enc_nonlinearmodel,dec_nonlinearmodel,LowRankMultivariateNormal)
    ###############################


    #### regularized nl model ####
    enc_nl_reg = ProbabilisticEncoder(n_layers_shared=4,n_layers_private=3,data_dim=proj_dim,hidden_dim=125,latent_dim=2)
    dec_nl_reg = SoftRegularizedDecoder(n_layers=7,n_layers=7,data_dim=proj_dim,hidden_dim=125,latent_dim=2)

    vae_nl_reg = VariationalAutoEncoder(enc_nl_reg,dec_nl_reg,LowRankMultivariateNormal)
    ###############################

    #### regularized nl lip mod ###
    enc_lip_reg = ProbabilisticEncoder(n_layers_shared=4,n_layers_private=3,data_dim=proj_dim,hidden_dim=125,latent_dim=2)
    dec_lip_reg = SoftRegularizedDecoder(n_layers=7,n_layers=7,data_dim=proj_dim,hidden_dim=125,latent_dim=2,max_lipschitz=l1_lip_proj,norm_func=l1norm)

    vae_lip_reg = VariationalAutoEncoder(enc_lip_reg,dec_lip_reg,LowRankMultivariateNormal)
    ################################

if __name__ == '__main__':

    fire.Fire(run_experiments)