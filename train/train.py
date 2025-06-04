import torch
from models.vae import *
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import copy

def save_model(model,optimizer,location):


    sd = {'ac': model.state_dict(),
          'opt':optimizer.state_dict(),
          'encoder specs': model.encoder.spec_dict,
          'decoder specs': model.decoder.spec_dict,
          'ac specs': model.spec_dict
    }

    torch.save(sd,location)

def load_model(location):

    sd = torch.load(location,weights_only=False)
    model_params = sd['ac']
    opt_params = sd['opt']
    encoder_details = sd['encoder specs']
    decoder_details = sd['decoder specs']
    ac_details = sd['ac specs']
    encoder_type = encoder_details['type']
    decoder_type = decoder_details['type']
    ac_type = ac_details['type']

    if encoder_type == 'MLP':
        enc = Encoder(n_layers=encoder_details['n_layers'],
                      data_dim=encoder_details['data_dim'],
                      hidden_dim=encoder_details['hidden_dim'],
                      latent_dim=encoder_details['latent_dim'],
                      activation=encoder_details['activation'],
                      device=encoder_details['device'])

    elif encoder_type == 'probabilistic':
        enc = ProbabilisticEncoder(n_layers_shared=encoder_details['n_layers_shared'],
                             n_layers_private=encoder_details['n_layers_private'],
                      data_dim=encoder_details['data_dim'],
                      hidden_dim=encoder_details['hidden_dim'],
                      latent_dim=encoder_details['latent_dim'],
                      activation=encoder_details['activation'],
                      device=encoder_details['device'])
    else:
        raise NotImplementedError
    
    if decoder_type == 'MLP':
        dec = Decoder(n_layers=decoder_details['n_layers'],
                      data_dim=decoder_details['data_dim'],
                      hidden_dim=decoder_details['hidden_dim'],
                      latent_dim=decoder_details['latent_dim'],
                      activation=decoder_details['activation'],
                      device=decoder_details['device'])
    elif decoder_type =='regularized MLP' :
        dec = RegularizedDecoder(n_layers=decoder_details['n_layers'],
                      data_dim=decoder_details['data_dim'],
                      hidden_dim=decoder_details['hidden_dim'],
                      latent_dim=decoder_details['latent_dim'],
                      activation=decoder_details['activation'],
                      device=decoder_details['device'])
    else:
        raise NotImplementedError
    
    if ac_type == 'autoencoder':
        model = AutoEncoder(enc,dec,device=ac_details['device'])

    elif ac_type =='VAE':
        model = VariationalAutoEncoder(enc,dec,device=ac_details['device'],latent_distribution=ac_details['latent_dist'])

    model.load_state_dict(model_params)
    opt=Adam(model.parameters(),lr=1e-3)
    opt.load_state_dict(opt_params)
    scheduler = ReduceLROnPlateau(opt,factor=0.75,patience=5,min_lr=1e-10)


    return model,opt,scheduler

def train(model,dataloaders,loss,regularizer = None, nEpochs=200,lr=1e-3,val_freq=10,vis_freq=1,max_norm_grad=1e-2,opt =None,start_epoch=0,save_freq=-1,model_prefix='model'):

    if opt == None:
        opt = Adam(model.parameters(),lr=lr)
    scheduler = ReduceLROnPlateau(opt,factor=0.75,patience=5,min_lr=1e-10)

    train_recon,val_recon,train_kl,val_kl,train_reg,val_reg = [],[],[],[],[],[]
    for epoch in tqdm(range(start_epoch,nEpochs+1),desc='training...'):

        model.train()

        for bi, batch in enumerate(dataloaders['train'],start=epoch*len(dataloaders['train'])):

            opt.zero_grad()
            batch = batch.to(model.device).to(torch.float32)

            model_out = model(batch)

            recon_loss, kl = loss(batch,model_out)

            l = recon_loss + kl

            if regularizer != None:
                reg = regularizer(model.decoder)
                l = l + reg
                train_reg.append(reg.item())
            else:
                train_reg.append(0.)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=max_norm_grad)
            opt.step()
            model.decoder.regularize()

            train_recon.append(recon_loss.item())
            train_kl.append(kl.item())
            

        if epoch % val_freq == 0:

            model.eval()

            vl,vk,vr = 0.,0.,0.
            for _, batch in enumerate(dataloaders['val']):
                batch=batch.to(model.device).to(torch.float32)

                model_out = model(batch)

                recon_loss, kl = loss(batch,model_out)

                vl += recon_loss.item()
                vk += kl.item()
                if regularizer != None:
                    reg = regularizer(model.decoder)
                    vr += reg.item()
            vl /= len(dataloaders['val'])
            vk /= len(dataloaders['val'])
            vr /= len(dataloaders['val'])
            val_recon.append((bi,vl))
            val_kl.append((bi,vk))
            val_reg.append((bi,vr))

            #l = val_recon + val_kl
            #scheduler.step(vl+vk)

        if (save_freq > 0) and (((epoch +1) % save_freq) == 0):
            save_model(model,opt,model_prefix + str(epoch) + '.tar')

        if epoch % vis_freq == 0:

            pass


    return model,opt,scheduler,(train_recon,val_recon),(train_kl,val_kl),(train_reg,val_reg)

def train_cv_reg(model,dataloaders,loss,regularizer, nEpochs=200,lr=1e-3,val_freq=10,vis_freq=1,max_norm_grad=1e-2,opt =None,start_epoch=0,save_freq=-1,model_prefix='model'):


    reg_weight_array = [1e-2,5e-2,1e-1,5e-1,1,5,10,50,100,500,1000,5000]
    model_copy = copy.deepcopy(model)

    final_elbos = []
    for reg_weight in reg_weight_array:

        reggie = lambda model: regularizer(model,weight=reg_weight)
        model_copy,temp_opt,scheduler,(train_recon,val_recon),(train_kl,val_kl), _ = train(model_copy,dataloaders=dataloaders,loss=loss,\
                                                                                 regularizer=reggie,nEpochs=nEpochs,lr=lr,val_freq=val_freq,\
                                                                                    vis_freq=vis_freq,max_norm_grad=max_norm_grad,opt=opt,\
                                                                                        start_epoch=start_epoch,save_freq=save_freq,model_prefix=model_prefix)
        final_elbo = -np.nanmean(np.array(val_recon)[:-10,1] - np.array(val_kl)[:-10,1])
        final_elbos.append(final_elbo)
    best_reg_weight = reg_weight_array[np.argmax(final_elbos)]
    best_reggie =  lambda model: regularizer(model,weight=best_reg_weight)

    model,opt,scheduler,(train_recon,val_recon),(train_kl,val_kl), (train_reg,val_reg) = train(model_copy,dataloaders=dataloaders,loss=loss,\
                                                                                 regularizer=best_reggie,nEpochs=nEpochs,lr=lr,val_freq=val_freq,\
                                                                                    vis_freq=vis_freq,max_norm_grad=max_norm_grad,opt=opt,\
                                                                                        start_epoch=start_epoch,save_freq=save_freq,model_prefix=model_prefix)
    
    return model,opt,scheduler,(train_recon,val_recon),(train_kl,val_kl),(train_reg,val_reg) 




    

    

    


