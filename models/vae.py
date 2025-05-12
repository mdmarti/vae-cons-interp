import torch
from torch import nn



class Encoder(nn.Module):


    def __init__(self,n_layers,data_dim,hidden_dim,latent_dim,activation=nn.GELU(),device='cuda'):

        super(Encoder,self).__init__()
        if n_layers == 0:
            self.net = nn.Linear(data_dim,latent_dim)
        else:
            layers = [nn.Linear(data_dim,hidden_dim), activation]
            for _ in range(n_layers - 1):
                layers += [nn.Linear(hidden_dim,hidden_dim),activation]
            layers += [nn.Linear(hidden_dim,latent_dim)]

            self.net = nn.Sequential(*layers)
        self.device = device
        self.to(device)
        self.spec_dict={
            'n_layers':n_layers,
            'data_dim':data_dim,
            'hidden_dim':hidden_dim,
            'latent_dim':latent_dim,
            'activation':activation,
            'device':device,
            'type':'MLP'
        }

    def forward(self,x):

        return self.net(x)
    
class ProbabilisticEncoder(Encoder):


    def __init__(self, n_layers_shared,n_layers_private,\
                 data_dim, hidden_dim, latent_dim, activation=nn.GELU(), device='cuda'):
        super().__init__(n_layers_shared, data_dim, hidden_dim, hidden_dim, activation, device)
        if n_layers_private == 0:
            self.mu_net = nn.Linear(hidden_dim,latent_dim)
            self.L_net = nn.Linear(hidden_dim,latent_dim)
            self.d_net = nn.Linear(hidden_dim,latent_dim)
        else:
            mu_layers = [nn.Linear(hidden_dim,hidden_dim),activation]
            L_layers = [nn.Linear(hidden_dim,hidden_dim),activation]
            d_layers = [nn.Linear(hidden_dim,hidden_dim),activation]
            for _ in range(n_layers_private - 1):
                mu_layers += [nn.Linear(hidden_dim,hidden_dim),activation]
                L_layers += [nn.Linear(hidden_dim,hidden_dim),activation]
                d_layers += [nn.Linear(hidden_dim,hidden_dim),activation]
            mu_layers += [nn.Linear(hidden_dim,latent_dim)]
            L_layers += [nn.Linear(hidden_dim,latent_dim)]
            d_layers += [nn.Linear(hidden_dim,latent_dim)]

            self.mu_net = nn.Sequential(*mu_layers)
            self.L_net = nn.Sequential(*L_layers)
            self.d_net = nn.Sequential(*d_layers)

        self.device = device
        self.to(device)
        self.spec_dict={
            'n_layers_shared':n_layers_shared,
            'n_layers_private':n_layers_private,
            'data_dim':data_dim,
            'hidden_dim':hidden_dim,
            'latent_dim':latent_dim,
            'activation':activation,
            'device':device,
            'type':'probabilistic'
        }

    def forward(self,x):

        x = self.net(x)
        
        mu,L,d = self.mu_net(x),self.L_net(x),self.d_net(x)

        return mu, L.unsqueeze(-1), d.exp()


class Decoder(nn.Module):


    def __init__(self,n_layers,data_dim,hidden_dim,latent_dim,activation=nn.GELU(),device='cuda'):

        super(Decoder,self).__init__()
        if n_layers == 0:
            self.net = nn.Linear(latent_dim,data_dim)
        else:
            layers = [nn.Linear(latent_dim,hidden_dim), activation]
            for _ in range(n_layers - 1):
                layers += [nn.Linear(hidden_dim,hidden_dim),activation]
            layers += [nn.Linear(hidden_dim,data_dim)]

            self.net = nn.Sequential(*layers)
        self.device = device
        self.to(device)
        self.spec_dict={
            'n_layers':n_layers,
            'data_dim':data_dim,
            'hidden_dim':hidden_dim,
            'latent_dim':latent_dim,
            'activation':activation,
            'device':device,
            'type': 'MLP'
        }


    def forward(self,x):
        return self.net(x)
    

class AutoEncoder(nn.Module):


    def __init__(self,encoder,decoder,device='cuda'):

        super(AutoEncoder,self).__init__()
        self.encoder=encoder 
        self.decoder=decoder
        self.device=device 
        self.to(device)

        self.spec_dict={
            'type':'autoencoder',
            'device':device
        }

    def forward(self,x):

        z = self.encoder(x)

        xhat = self.decoder(z)

        return xhat,z 
    
    def encode(self,x):

        return self.encoder(x)
    
    def decode(self,z):

        return self.decoder(z)
    
class VariationalAutoEncoder(AutoEncoder):

    def __init__(self, encoder, decoder, latent_distribution, device='cuda'):
        super().__init__(encoder, decoder, device)

        self.latent_distribution=latent_distribution
        self.spec_dict={
            'type':'VAE',
            'device':device,
            'latent_dist':latent_distribution
        }


    def forward(self,x):


        latent_params = self.encoder(x)
        latent_dist = self.latent_distribution(*latent_params)

        z = latent_dist.rsample()

        xhat = self.decoder(z)

        return xhat,z,latent_dist







