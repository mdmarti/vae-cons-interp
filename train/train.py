import torch
from models.vae import *
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

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

def train(model,dataloaders,loss,nEpochs=200,lr=1e-3,val_freq=10,vis_freq=1):


    opt = Adam(model.parameters(),lr=lr)
    scheduler = ReduceLROnPlateau(opt,factor=0.75,patience=5,min_lr=1e-10)

    train_recon,val_recon,train_reg,val_reg = [],[],[],[]
    for epoch in tqdm(range(nEpochs),desc='training...'):

        model.train()

        for bi, batch in enumerate(dataloaders['train'],start=epoch*len(dataloaders['train'])):

            opt.zero_grad()
            batch = batch.to(model.device).to(torch.float32)

            model_out = model(batch)

            recon_loss, latent_reg = loss(batch,model_out)

            l = recon_loss + latent_reg

            
            l.backward()
            opt.step()

            train_recon.append(recon_loss.item())
            train_reg.append(latent_reg.item())

        if epoch % val_freq == 0:

            model.eval()

            vl,vr = 0.,0.
            for _, batch in enumerate(dataloaders['val']):
                batch=batch.to(model.device).to(torch.float32)

                model_out = model(batch)

                recon_loss, latent_reg = loss(batch,model_out)

                vl += recon_loss.item()
                vr += latent_reg.item()
            vl /= len(dataloaders['val'])
            vr /= len(dataloaders['val'])
            val_recon.append((bi,vl))
            val_reg.append((bi,vr))

            l = val_recon + val_reg
            scheduler.step(vl+vr)

        if epoch % vis_freq == 0:

            pass


    return model,opt,scheduler,(train_recon,val_recon),(train_reg,val_reg)



    

    

    


