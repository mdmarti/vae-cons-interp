import torch.nn as nn
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
import numpy as np
from data.toy_data import *
from data.data_utils import *
from train.losses import *
from train.train import train,save_model,load_model

from models.vae import *
from eval.metrics import assess_gmm_fit,get_all_stats
import os
#from fire import fire

from sklearn.mixture import GaussianMixture as GMM
from eval.eval import train_test_plot,embedding_plot
import json
import glob
from visualization.toy_data_plots import make_toy_plot
import fire

def find_create_model(target_model_prefix):
    #target_fp = target_model_path.split('/')[-1]
    #target_prefix = target_fp.split('_*precision*.tar')[0]
    current_matching_model_files = glob.glob(target_model_prefix + '*.tar')
    if len(current_matching_model_files) == 0:
        return None,None,1

    save_epochs = [int(fp.split('checkpoint_')[-1].split('.tar')[0]) for fp in current_matching_model_files]
    file_order = np.argsort(save_epochs)
    max_epoch = save_epochs[file_order[-1]]
    most_recent_model=current_matching_model_files[file_order[-1]]

    
    vae,opt,scheduler = load_model(most_recent_model)
    return vae,opt, max_epoch

def run_helper(save_dir,model_type,precision,loaders,data,labels,nEpochs=1000,lr=1e-3,
               n_layers_shared=4,n_layers_private=3,data_dim=1000,hidden_dim=125,latent_dim=2,
               device='default',
               n_layers_decoder=7,decoder_activation=nn.GELU()):

    model_path = os.path.join(save_dir,f'vae_{model_type}decoder_{precision}precision_checkpoint_{nEpochs}.tar')
    model_prefix = model_path.split('.tar')[0]
    
    train_stats_path = os.path.join(save_dir,f'train_stats_{model_type}_{precision}.json')
    embed_path = os.path.join(save_dir,f'embeddings_recons_{model_type}_{precision}.json')
    
    done_training = False
    n_attempts = 0

    loss = lambda target, model_out: ELBO_more_stable(target,model_out,recon_precision=precision)
    if not os.path.isfile(model_path):
        while n_attempts < 5 and not done_training:

            try:
                vae,opt,start_epoch = find_create_model(model_prefix)
                if start_epoch >= nEpochs:
                    done_training = True
                    with open(train_stats_path,'r') as f:
                        train_stats = json.load(f)
                    log_probs=train_stats['lp']
                    kls=train_stats['kl']
                            
                    with open(embed_path,'r') as f:
                        model_outputs = json.load(f)
                    embeddings,recons=np.array(model_outputs['embeddings']),np.array(model_outputs['recons'])
                    train_test_plot(log_probs,kls,label=f"{model_type} decoder, recon precision = {precision}",show=False,\
                                save_fn=os.path.join(save_dir,f'{model_type}_{precision}_traintest.svg'))
                    embedding_plot(embeddings,recons,data,labels,label=f"{model_type} decoder, recon precision = {precision}",show=False,\
                                    save_fn=os.path.join(save_dir,f'{model_type}_{precision}_embeds.svg'))

                    return log_probs,kls,embeddings,recons

                if start_epoch == 1:
                    enc = ProbabilisticEncoder(n_layers_shared=n_layers_shared,n_layers_private=n_layers_private,
                                            data_dim=data_dim,hidden_dim=hidden_dim,latent_dim=latent_dim,device=device)
                    
                    if model_type == 'regularized_nonlinear':
                        dec = RegularizedDecoder(n_layers=n_layers_decoder,data_dim=data_dim,hidden_dim=hidden_dim,latent_dim=latent_dim,
                                    activation=decoder_activation,device=device)
                        loss = lambda target,model_out : ELBO_linear_encouragement(target,model_out,dec,recon_precision=precision,weight_penalty=1) ### try this out on monday
                    else:
                        dec = Decoder(n_layers=n_layers_decoder,data_dim=data_dim,hidden_dim=hidden_dim,latent_dim=latent_dim,
                                    activation=decoder_activation,device=device)
                        
                    vae = VariationalAutoEncoder(enc,dec,LowRankMultivariateNormal,out_type='params')
                    opt=None
                vae,opt,scheduler,log_probs,kls = train(vae,loaders,loss=loss,
                                                    nEpochs=nEpochs,val_freq=10,lr=lr,max_norm_grad=1e-2,start_epoch=start_epoch,opt =opt,save_freq=100,model_prefix=model_prefix)
                done_training=True 
            except:
                print("bad params, restarting")
                n_attempts += 1
        if done_training:
            save_model(vae,opt,model_path)
            embeddings = vae.encode(torch.from_numpy(data).to(vae.device).to(torch.float32))[0]
            recons= vae.decode(embeddings).detach().cpu().numpy()
            embeddings = embeddings.detach().cpu().numpy()

            train_stats = {'lp':log_probs,'kl':kls}
            model_outputs = {'embeddings':embeddings.tolist(),'recons':recons.tolist()}

            with open(train_stats_path,'w') as f:
                json.dump(train_stats,f)
            with open(embed_path,'w') as f:
                json.dump(model_outputs,f)

    else:
        vae,opt,scheduler = load_model(model_path)
    
        with open(train_stats_path,'r') as f:
            train_stats = json.load(f)
        log_probs=train_stats['lp']
        kls=train_stats['kl']
                
        with open(embed_path,'r') as f:
            model_outputs = json.load(f)
        embeddings,recons=np.array(model_outputs['embeddings']),np.array(model_outputs['recons'])
        done_training=True

    if done_training:
        train_test_plot(log_probs,kls,label=f"{model_type} decoder, recon precision = {precision}",show=False,\
                    save_fn=os.path.join(save_dir,f'{model_type}_{precision}_traintest.svg'))
        embedding_plot(embeddings,recons,data,labels,label=f"{model_type} decoder, recon precision = {precision}",show=False,\
                        save_fn=os.path.join(save_dir,f'{model_type}_{precision}_embeds.svg'))

        return log_probs,kls,embeddings,recons
    
    return [],[],[],[]

def run_experiments(save_dir,n_samples=15000,proj_dim = 1000,nEpochs=1000,linear=True,identity=False,seed=99,proj_sd=0.08):

    n_workers = len(os.sched_getaffinity(0))
    if identity:

        proj = IdentityProjection(data_dim=2)

    elif linear:

        proj = LinearProjection(data_dim=2,project_dim=proj_dim,seed=seed)

    else:

        nonlinearity = lambda x: np.log(x + 1) if x > 0 else -np.log(-x + 1)
        inverse_nonlinearity = lambda x: np.exp(x) - 1 if x > 0 else -np.exp(-x) + 1 

        proj = NonlinearProjection(data_dim=2,project_dim=proj_dim,\
                                   nonlinearity=nonlinearity,inverse_nonlinearity=inverse_nonlinearity,seed=seed)


    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    precision_pts = np.array([1] + list(10*np.arange(1,11)))
    latents,data,labels,med_dist = generate_mixture_dataset(n_samples=n_samples,projection=proj,proj_sd=proj_sd,seed=seed)

    closest_pt = precision_pts[np.argmin(np.abs(precision_pts - med_dist))]
    l1_lip_proj = l1norm(torch.from_numpy(proj.w))

    base_model = GMM(n_components=4,covariance_type='full',n_init=10)
    pred_labels = base_model.fit_predict(data)
    #print(pred_labels.shape)
    #print(labels.shape)
    base_precisions,base_recalls = assess_gmm_fit(labels,pred_labels)

    loaders = get_loaders(data,test_size=0.4,seed=seed,num_workers = 8,batch_size=512)

    precisions = np.array([closest_pt/4,closest_pt/2,closest_pt,closest_pt*2,closest_pt*4])*100 #np.logspace(-2,3,1)
    lr = 1e-3
    for p in precisions:

        print(f'now fitting for precision = {p}')
        ##### Linaer model #######
        vae_linear_lps,vae_linear_kls,vae_linearlatents,vae_linearrecons = run_helper(save_dir,model_type='linear',precision=p,\
                                                           loaders=loaders,data=data,labels=labels,nEpochs=nEpochs,lr=lr,\
                                                            n_layers_shared=4,n_layers_private=3,data_dim=proj_dim,hidden_dim=125,latent_dim=2,device='default',\
                                                                n_layers_decoder=0,decoder_activation=nn.Identity())

        linear_gmm = GMM(n_components=4,covariance_type='full',n_init=10)
        pred_labels_linear = linear_gmm.fit_predict(vae_linearlatents)
        #linear_model_metrics = get_all_stats(latents,vae_linearlatents,labels,pred_labels_linear)
        #############################

        #### deep linear model ######
        vae_deeplinear_lps,vae_deeplinear_kls,vae_deeplinearlatents,vae_deeplinearrecons = run_helper(save_dir,model_type='deeplinear',precision=p,\
                                                        loaders=loaders,data=data,labels=labels,nEpochs=nEpochs,lr=lr,\
                                                            n_layers_shared=4,n_layers_private=3,data_dim=proj_dim,hidden_dim=125,latent_dim=2,device='default',\
                                                                n_layers_decoder=7,decoder_activation=nn.Identity())


        

        deeplinear_gmm = GMM(n_components=4,covariance_type='full',n_init=10)
        pred_labels_deeplinear = deeplinear_gmm.fit_predict(vae_deeplinearlatents)
        #deeplinear_model_metrics = get_all_stats(latents,vae_deeplinearlatents,labels,pred_labels_deeplinear)
        
        ##############################


        #### nonlinear model #########
        vae_nonlinear_lps,vae_nonlinear_kls,vae_nonlinearlatents,vae_nonlinearrecons = run_helper(save_dir,model_type='nonlinear',precision=p,\
                                                        loaders=loaders,data=data,labels=labels,nEpochs=nEpochs,lr=lr,\
                                                            n_layers_shared=4,n_layers_private=3,data_dim=proj_dim,hidden_dim=125,latent_dim=2,device='default',\
                                                                n_layers_decoder=7,decoder_activation=nn.GELU())
        
        nonlinear_gmm = GMM(n_components=4,covariance_type='full',n_init=10)
        pred_labels_nonlinear = nonlinear_gmm.fit_predict(vae_nonlinearlatents)
        #nonlinear_model_metrics = get_all_stats(latents,vae_nonlinearlatents,labels,pred_labels_nonlinear)

        ##############################
        """
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
        """

        make_toy_plot(latents,data,vae_linearlatents,vae_linearrecons,\
                      vae_deeplinearlatents,vae_deeplinearrecons,\
                        vae_nonlinearlatents,vae_nonlinearrecons,labels,\
                            show=False,save_fn=os.path.join(save_dir,f'all_models_plot_{p}.svg'))

if __name__ == '__main__':

    fire.Fire(run_experiments)