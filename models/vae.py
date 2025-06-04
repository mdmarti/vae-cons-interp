import torch
from torch import nn



class Encoder(nn.Module):


    def __init__(self,n_layers,data_dim,hidden_dim,latent_dim,activation=nn.GELU(),device='default'):

        super(Encoder,self).__init__()
        if n_layers == 0:
            self.net = nn.Linear(data_dim,latent_dim)
        else:
            layers = [nn.Linear(data_dim,hidden_dim), activation]
            for _ in range(n_layers - 1):
                layers += [nn.Linear(hidden_dim,hidden_dim),activation]
            layers += [nn.Linear(hidden_dim,latent_dim)]

            self.net = nn.Sequential(*layers)
        if device == 'default':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
                 data_dim, hidden_dim, latent_dim, activation=nn.GELU(), device='default'):
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
        if device == 'default':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
            'type':'probabilistic',
        }

    def forward(self,x):

        x = self.net(x)
        
        mu,L,d = self.mu_net(x),self.L_net(x),self.d_net(x)

        return mu, L.unsqueeze(-1), d.exp()
    
class RegularizedProbabilisticEncoder(Encoder):


    def __init__(self, n_layers_shared,n_layers_private,\
                 data_dim, hidden_dim, latent_dim,layer_type, activation=nn.GELU(), device='default'):
        super().__init__( n_layers_shared,n_layers_private,\
                 data_dim, hidden_dim, latent_dim, activation=nn.GELU(), device='default')
        if n_layers_private == 0:
            self.mu_net = nn.Linear(hidden_dim,latent_dim)
            self.L_net = nn.Linear(hidden_dim,latent_dim)
            self.d_net = nn.Linear(hidden_dim,latent_dim)
        else:
            mu_layers = [layer_type(latent_dim,hidden_dim,activation)]
            L_layers = [layer_type(latent_dim,hidden_dim,activation)]
            d_layers = [layer_type(latent_dim,hidden_dim,activation)]
            for _ in range(n_layers_private - 1):
                mu_layers += [layer_type(latent_dim,hidden_dim,activation)]
                L_layers += [layer_type(latent_dim,hidden_dim,activation)]
                d_layers += [layer_type(latent_dim,hidden_dim,activation)]
            mu_layers += [nn.Linear(hidden_dim,latent_dim)]
            L_layers += [nn.Linear(hidden_dim,latent_dim)]
            d_layers += [nn.Linear(hidden_dim,latent_dim)]

            self.mu_net = nn.Sequential(*mu_layers)
            self.L_net = nn.Sequential(*L_layers)
            self.d_net = nn.Sequential(*d_layers)
        if device == 'default':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
            'type':'probabilistic',
        }

    def forward(self,x):

        x = self.net(x)
        
        mu,L,d = self.mu_net(x),self.L_net(x),self.d_net(x)

        return mu, L.unsqueeze(-1), d.exp()
        
class LinearEncouragementLayer(nn.Module):

    def __init__(self,in_size,out_size,activation,device='default',full_prelu=False):

        super(LinearEncouragementLayer,self).__init__()
        if device == 'default':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.linear = nn.Linear(in_size,out_size,device=device)
        if full_prelu:
            self.nonlinearity =nn.PReLU(num_parameters =out_size) #out_size -- one extra parameter per layer might be easier to fit
        else:
            self.nonlinearity =nn.PReLU(num_parameters =1)

    def forward(self,x):
        return self.nonlinearity(self.linear(x))
    
class LinearEncouragementLayer_v2(nn.Module):

    def __init__(self,in_size,out_size,activation,device='default'):

        super(LinearEncouragementLayer_v2,self).__init__()

        if device == 'default':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.linear = nn.Linear(in_size,out_size,device=device)
        self.nonlinearity = nn.Linear(in_size,out_size,device=device)
        self.activation=activation 

    def forward(self,x):

        return self.linear(x) + self.activation(self.nonlinearity(x))
    
class LipschitzPlusUnCon(nn.Module):

    def __init__(self,in_size,out_size,nonlinearity=nn.GELU(),device='default',\
                 norm_func = lambda w: power_iter(w,n_power_iters=10),lip_const=1.5):

        super(LipschitzPlusUnCon,self).__init__()
        if device == 'default':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lipschitz = nn.Linear(in_size,out_size,device=device)
        self.unconstrained = nn.Linear(in_size,out_size,device=device)
        self.nonlinearity=nonlinearity
        self.lip_const=lip_const
        self.norm_func=norm_func


    def forward(self,x,return_both=False):

        if return_both:

            return self.nonlinearity(self.lipschitz(x)),self.nonlinearity(self.unconstrained(x))
        
        else:

            return self.nonlinearity(self.lipschitz(x)) + self.nonlinearity(self.unconstrained(x))


    def regularize(self):
        self.lipschitz_constrain()

    def lipschitz_constrain(self):

        #pre_weights = []
        w = self.lipschitz.weight
        operator_norm = self.norm_func(w)
        self.lipschitz.weight /= max(1,operator_norm/self.lip_const)


#### finish this here


class Decoder(nn.Module):


    def __init__(self,n_layers,data_dim,hidden_dim,latent_dim,activation=nn.GELU(),device='default'):

        super(Decoder,self).__init__()
        if n_layers == 0:
            self.net = nn.Linear(latent_dim,data_dim)
        else:
            layers = [nn.Linear(latent_dim,hidden_dim), activation]
            for _ in range(n_layers - 1):
                layers += [nn.Linear(hidden_dim,hidden_dim),activation]
            layers += [nn.Linear(hidden_dim,data_dim)]

            self.net = nn.Sequential(*layers)
        if device == 'default':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    
    def regularize(self):
        pass

class RegularizedDecoder(Decoder):

    def __init__(self, n_layers, data_dim, hidden_dim, latent_dim,layer_type, activation=nn.GELU(),device='default'):
        super().__init__(n_layers, data_dim, hidden_dim, latent_dim, activation, device)

        if n_layers == 0:
            self.net = layer_type(latent_dim,data_dim)
        else:
            layers = [layer_type(latent_dim,hidden_dim,activation)]
            for _ in range(n_layers - 1):
                layers += [layer_type(hidden_dim,hidden_dim,activation)]
            layers += [nn.Linear(hidden_dim,data_dim)]

            self.net = nn.Sequential(*layers)
        if device == 'default':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.to(device)
        self.spec_dict={
            'n_layers':n_layers,
            'data_dim':data_dim,
            'hidden_dim':hidden_dim,
            'latent_dim':latent_dim,
            'activation':nn.Identity(),
            'device':device,
            'type': 'regularized MLP'
        }


def power_iter(weight,n_power_iters):

    x = torch.randn((weight.shape[1]),device=weight.device)
    for ii in range(n_power_iters):
        x = weight.T @ weight @ x
    
    return torch.linalg.norm(weight @ x)/torch.linalg.norm(x)  

def l1norm(weight):

    return torch.amax(torch.sum(weight.abs(),dim=0))

def linfnorm(weight):

    return torch.amax(torch.sum(weight.abs(),dim=1))


class LipschitzDecoder(Decoder):

    def __init__(self,n_layers,data_dim,hidden_dim,latent_dim,activation=nn.GELU(),device='default',max_lipschitz=2,
                 norm_func=lambda w: power_iter(w,n_power_iters=10)):

        super(LipschitzDecoder,self).__init__(n_layers,data_dim,hidden_dim,latent_dim,activation,device)
        self.max_lipschitz=max_lipschitz
        self.norm_func=norm_func

    def regularize(self):
        self.lipschitz_constrain()

    def lipschitz_constrain(self):

        #pre_weights = []
        for name,module in self.named_modules():

            try:
                w = module.weight
            except:
                continue 
            #pre_weights.append(w.detach().cpu().numpy())
            operator_norm = self.norm_func(w)
            w = w / max(1,operator_norm/self.max_lipschitz)
            with torch.no_grad():
                module.weight /= max(1,operator_norm/self.max_lipschitz)
            if operator_norm > self.max_lipschitz:
                print(f"constraining {name}")

    

class AutoEncoder(nn.Module):


    def __init__(self,encoder,decoder,device='default'):

        super(AutoEncoder,self).__init__()
        self.encoder=encoder 
        self.decoder=decoder
        if device == 'default':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    

VALID_OUT_TYPES=['dist','params']
class VariationalAutoEncoder(AutoEncoder):

    def __init__(self, encoder, decoder, latent_distribution, device='default',out_type='dist'):
        super().__init__(encoder, decoder, device)

        self.latent_distribution=latent_distribution
        assert out_type in VALID_OUT_TYPES,print(f"Output type of VAE must be in {VALID_OUT_TYPES},you chose {out_type}")
        self.out_type = out_type
        self.spec_dict={
            'type':'VAE',
            'device':device,
            'latent_dist':latent_distribution,
            'output_type':self.out_type
        }


    def forward(self,x):


        latent_params = self.encoder(x)
        latent_dist = self.latent_distribution(*latent_params)

        z = latent_dist.rsample()

        xhat = self.decoder(z)

        if self.out_type == 'dist':
            return xhat,z,latent_dist
        elif self.out_type == 'params':
            return xhat,z,latent_params








