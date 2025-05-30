import numpy as np 

class LinearProjection():

    def __init__(self,data_dim,project_dim):

        self.data_dim = data_dim
        self.project_dim=project_dim
        self.w = np.random.randn(data_dim,project_dim)
        #self.noise_sd = noise_sd

    def __call__(self,data,noise_sd=0.):

        return self.project(data,noise_sd)

    def project(self,data,noise_sd=0.):

        # expects data to be N x d, projection weight to be d x P

        return data @ self.w + noise_sd * np.random.randn(data.shape[0],self.project_dim)

    def un_project(self,data):

        return data @ self.w.T @ np.linalg.pinv(self.w @ self.w.T)
    
class IdentityProjection(LinearProjection):

    def __init__(self,data_dim):

        super(IdentityProjection,self).__init__(data_dim,data_dim)

        self.w = np.eye(data_dim)
    
class NonlinearProjection(LinearProjection):

    def __init__(self,data_dim,project_dim,nonlinearity,inverse_nonlinearity):
        ### nonlinearity must be invertible

        super(NonlinearProjection,self).__init__(data_dim=data_dim,project_dim=project_dim)
        self.nonlinearity = nonlinearity
        self.inverse_nonlinearity=inverse_nonlinearity

    def project(self,data,noise_sd=0.):

        return self.nonlinearity(super().project(data,noise_sd=0.)) + noise_sd * np.random.randn(data.shape[0],self.project_dim)
    
    def un_project(self, data):
        return super().un_project(self.inverse_nonlinearity(data))

def generate_mixture_dataset(projection,n_samples=100,proj_sd=0.,seed=None):

    ## play around with parameters here
    generator = np.random.default_rng(seed=seed)
    
    mixture_weights = [0.25,0.1,0.3,0.35]
    mu1,mu2,mu3,mu4 = np.array([-0.05,1]),np.array([1,0]),np.array([2,0]),np.array([-0.5,-0.25])
    cov1,cov2,cov3,cov4 = np.array([[1,0],[0,1]]),np.array([[1,0.35],[0.35,1]]),np.array([[1,0.25],[0.25,1]]),np.array([[1,-0.75],[-0.75,1]])

    mus = np.stack([mu1,mu2,mu3,mu4],axis=0)
    covs = np.stack([cov1,cov2,cov3,cov4],axis=0)/5

    sample_labels = generator.choice(4,n_samples,replace=True,p=mixture_weights)
    #print(samples.shape)
    #print(np.sum(sample_labels == 0)/n_samples)
    #print(np.sum(sample_labels == 1)/n_samples)
    #print(np.sum(sample_labels == 2)/n_samples)
    #print(np.sum(sample_labels == 3)/n_samples)

    mu_samples,cov_samples = mus[sample_labels],covs[sample_labels]

    data_samples = generator.multivariate_normal(mean=np.zeros((2,)),cov=np.eye(2),size=(n_samples))
    #print(data_samples.shape)
    data_samples = mu_samples + np.einsum('nkp,np->nk',cov_samples,data_samples)

    projected_samples = projection(data_samples,proj_sd)
    #print(data_samples.shape)
    return data_samples,projected_samples,sample_labels
