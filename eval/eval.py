import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def train_test_plot(recons,regularization,label=''):

    (train_recons,test_recons) = recons
    train_recons,test_recons=np.array(train_recons),np.array(test_recons)
    (train_kls,test_kls) = regularization
    train_kls,test_kls=np.array(train_kls),np.array(test_kls)
    
    fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
    ax1.plot(train_recons)
    ax1.plot(test_recons[:,0],test_recons[:,1])
    ax1.set_title("negative log probability")
    #ax.set_yscale('log')
    
    ax2.plot(train_kls)
    ax2.plot(test_kls[:,0],test_kls[:,1])
    #ax.set_yscale('log')
    ax2.set_title("KL term")
    fig.suptitle(label)
    plt.tight_layout()
    plt.show()
    plt.close()

def embedding_plot(embeddings,reconstructions,original_embeddings,data_labels,label=''):

    
    unprojected = PCA(n_components=2).fit_transform(reconstructions) #original_projection.un_project(reconstructions)
    
    #embeddings= embeddings.detach().cpu().numpy()
    #print(embeddings.shape,data.shape)
    fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(15,5))
    for ii in range(4):
        inds = data_labels == ii
        ax1.scatter(embeddings[inds,0],embeddings[inds,1],label=f'cluster {ii+1}')
    
    ax1.legend()
    ax1.set_title(f"VAE embeddings")
    
    for ii in range(4):
        inds = data_labels == ii
        ax2.scatter(unprojected[inds,0],unprojected[inds,1],label=f'cluster {ii+1}')
    
    ax2.legend()
    ax2.set_title(f"Unprojected VAE reconstructions")

    for ii in range(4):
        inds = data_labels == ii
        ax3.scatter(original_embeddings[inds,0],original_embeddings[inds,1],label=f'cluster {ii+1}')
    
    ax3.legend()
    ax3.set_title(f"Original embeddings")
    fig.suptitle(label)
    plt.tight_layout()
    plt.show()
    plt.close()

    