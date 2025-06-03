import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from visualization.utils import format_axis

def train_test_plot(recons,regularization,label='',show=False,save_fn='train_test_curves.svg'):

    (train_recons,test_recons) = recons
    train_recons,test_recons=np.array(train_recons),np.array(test_recons)
    (train_kls,test_kls) = regularization
    train_kls,test_kls=np.array(train_kls),np.array(test_kls)
    
    fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
    ax1.plot(train_recons[:,0],label='Train')
    ax1.plot(test_recons[:,0],test_recons[:,1],label='Validation')

    #xlim,ylim = ax1.get_xlim(),ax1.get_ylim()
    # np.arange(xlim[0],xlim[1]+1,(xlim[1]-xlim[0]) //5),np.arange(ylim[0],ylim[1]+1,(xlim[1]-ylim[0]) //5)
    ax1 = format_axis(ax1,xlabel='Gradient steps',ylabel='Negative log probability',
                      xticks=ax1.get_xticks(),
                      yticks=ax1.get_yticks(),
                      xlims=ax1.get_xlim(),
                      ylims=ax1.get_ylim())
    #ax1.set_title("negative log probability")
    
    #ax.set_yscale('log')
    
    ax2.plot(train_kls[:,0])
    ax2.plot(test_kls[:,0],test_kls[:,1])
    ax2 = format_axis(ax2,xlabel='Gradient steps',ylabel='KL Divergence',
                      xticks=ax2.get_xticks(),
                      yticks=ax2.get_yticks(),
                      xlims=ax2.get_xlim(),
                      ylims=ax2.get_ylim())
    #ax.set_yscale('log')
    #ax2.set_title("KL term")
    ax1.legend()
    fig.suptitle(label)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(save_fn)
    plt.close()

def embedding_plot(embeddings,reconstructions,original_embeddings,data_labels,label='',show=False,save_fn='embeddings_recons.svg'):

    
    unprojected = PCA(n_components=2).fit_transform(reconstructions) #original_projection.un_project(reconstructions)
    
    #embeddings= embeddings.detach().cpu().numpy()
    #print(embeddings.shape,data.shape)
    fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(15,5))
    for ii in range(4):
        inds = data_labels == ii
        ax1.scatter(embeddings[inds,0],embeddings[inds,1],label=f'cluster {ii+1}')
    
    ax1.legend()
    ax1 = format_axis(ax1,xlabel='VAE Latent 1',ylabel='VAE Latent 2',title='VAE embeddings',
                      xticks=ax1.get_xticks(),
                      yticks=ax1.get_yticks(),
                      xlims=ax1.get_xlim(),
                      ylims=ax1.get_ylim())
    ax1.set_title(f"VAE embeddings")
    
    for ii in range(4):
        inds = data_labels == ii
        ax2.scatter(unprojected[inds,0],unprojected[inds,1],label=f'cluster {ii+1}')
    
    ax2 = format_axis(ax2,xlabel='Data dim 1',ylabel='Data dim 2',title='VAE reconstructions',
                      xticks=ax2.get_xticks(),
                      yticks=ax2.get_yticks(),
                      xlims=ax2.get_xlim(),
                      ylims=ax2.get_ylim())

    for ii in range(4):
        inds = data_labels == ii
        ax3.scatter(original_embeddings[inds,0],original_embeddings[inds,1],label=f'cluster {ii+1}')
    
    #ax3.legend()
    ax3 = format_axis(ax3,xlabel='Data latent 1',ylabel='Data latent 2',title='Data latents',
                      xticks=ax3.get_xticks(),
                      yticks=ax3.get_yticks(),
                      xlims=ax3.get_xlim(),
                      ylims=ax3.get_ylim())
    #ax3.set_title(f"Original embeddings")
    fig.suptitle(label)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(save_fn)
    plt.close()

    