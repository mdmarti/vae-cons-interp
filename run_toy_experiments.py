import torch.nn as nn
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
import numpy as np
from data.toy_data import *
from data.data_utils import *
from train.losses import *
from train.train import train,save_model

from models.vae import *
from eval.metrics import assess_gmm_fit,get_all_stats
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

    base_precisions,base_recalls = assess_gmm_fit(labels,pred_labels)

    loaders = get_loaders(data,test_size=0.4,seed=777,num_workers = os.cpu_count()//2,batch_size=512)

    precisions = np.logspace(-2,3,10)
    lr = 1e-3
    for p in precisions:

        loss = lambda target, model_out: ELBO(target,model_out,recon_precision=precisions[0])
        ### simple linear model #####
        
        done_training = False
        n_attempts = 0
        while n_attempts < 5 and not done_training:
            try:
                enc_linearmodel = ProbabilisticEncoder(n_layers_shared=4,n_layers_private=3,data_dim=proj_dim,hidden_dim=125,latent_dim=2,device='default')
                dec_linearmodel = Decoder(n_layers=0,data_dim=proj_dim,hidden_dim=0,latent_dim=2,activation=nn.Identity())

                vae_linearmodel = VariationalAutoEncoder(enc_linearmodel,dec_linearmodel,LowRankMultivariateNormal,out_type='params')

                vae_linearmodel,opt_linearmodel,scheduler_linearmodel,reconstructions_linearmodel,regs_linearmodel=train(vae_linearmodel,loaders,loss=loss,\
                                                                                                                        nEpochs=1000,val_freq=10,lr=lr,max_norm_grad=1e-2)
                done_training=True
            except:
                n_attempts += 1
                print("bad params, restarting")
        
        vae_linearlatents = vae_linearmodel.encode(torch.from_numpy(data).to(vae_linearmodel.device).to(torch.float32))[0]
        linear_gmm = GMM(n_components=4,covariance_type='full',n_init=10)
        pred_labels_linear = linear_gmm.fit_predict(vae_linearlatents)
        linear_model_metrics = get_all_stats(latents,vae_linearlatents,labels,pred_labels_linear)
        save_model(vae_linearmodel,opt_linearmodel,os.path.join(save_dir,f'vae_lineardecoder_{p}precision_final.tar'))
        #############################

        #### deep linear model ######

        n_attempts = 0
        done_training=False
        while n_attempts < 5 and not done_training:
            try:
                enc_deeplinearmodel = ProbabilisticEncoder(n_layers_shared=4,n_layers_private=3,data_dim=proj_dim,hidden_dim=125,latent_dim=2)
                dec_deeplinearmodel = Decoder(n_layers=7,data_dim=proj_dim,hidden_dim=125,latent_dim=2,activation=nn.Identity())

                vae_deeplinearmodel = VariationalAutoEncoder(enc_deeplinearmodel,dec_deeplinearmodel,LowRankMultivariateNormal,out_type='params')
                vae_deeplinearmodel,opt_deeplinearmodel,scheduler_deeplinearmodel,reconstructions_deeplinearmodel,regs_deeplinearmodel=train(vae_deeplinearmodel,loaders,loss=loss,\
                                                                                                                    nEpochs=1000,val_freq=10,lr=lr,max_norm_grad=1e-2)
                done_training=True

            except:
                n_attempts += 1
                print("bad params, restarting")
        
        vae_deeplinearlatents = vae_deeplinearmodel.encode(torch.from_numpy(data).to(vae_deeplinearmodel.device).to(torch.float32))[0]
        deeplinear_gmm = GMM(n_components=4,covariance_type='full',n_init=10)
        pred_labels_deeplinear = deeplinear_gmm.fit_predict(vae_deeplinearlatents)
        deeplinear_model_metrics = get_all_stats(latents,vae_deeplinearlatents,labels,pred_labels_deeplinear)
        save_model(vae_deeplinearmodel,opt_deeplinearmodel,os.path.join(save_dir,f'vae_deeplineardecoder_{p}precision_final.tar'))
        ##############################


        #### nonlinear model #########
        n_attempts = 0
        done_training=False
        while n_attempts < 5 and not done_training:
            try:
                enc_nonlinearmodel = ProbabilisticEncoder(n_layers_shared=4,n_layers_private=3,data_dim=proj_dim,hidden_dim=125,latent_dim=2)
                dec_nonlinearmodel = Decoder(n_layers=7,data_dim=proj_dim,hidden_dim=125,latent_dim=2,activation=nn.GELU())

                vae_nonlinearmodel = VariationalAutoEncoder(enc_nonlinearmodel,dec_nonlinearmodel,LowRankMultivariateNormal,out_type='params')


                vae_nonlinearmodel,opt_nonlinearmodel,scheduler_nonlinearmodel,reconstructions_nonlinearmodel,regs_nonlinearmodel=train(vae_nonlinearmodel,loaders,loss=loss,\
                                                                                                                        nEpochs=1000,val_freq=10,lr=lr,max_norm_grad=1e-2)
                done_training=True

            except:
                n_attempts += 1
                print("bad params, restarting")
        
        vae_nonlinearlatents = vae_nonlinearmodel.encode(torch.from_numpy(data).to(vae_nonlinearmodel.device).to(torch.float32))[0]
        nonlinear_gmm = GMM(n_components=4,covariance_type='full',n_init=10)
        pred_labels_nonlinear = nonlinear_gmm.fit_predict(vae_nonlinearlatents)
        nonlinear_model_metrics = get_all_stats(latents,vae_nonlinearlatents,labels,pred_labels_nonlinear)
        save_model(vae_nonlinearmodel,opt_nonlinearmodel,os.path.join(save_dir,f'vae_nonlineardecoder_{p}precision_final.tar'))

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