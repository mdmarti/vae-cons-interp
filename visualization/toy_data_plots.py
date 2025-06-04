import numpy as np
import matplotlib.pyplot as plt
from visualization.utils import format_axis
from sklearn.decomposition import PCA


def make_toy_plot(true_latents,train_data,\
                  linear_latents,linear_recons,\
                    deep_linear_latents,deep_linear_recons,\
                        nonlinear_latents,nonlinear_recons,
                        nonlinear_prelureg_latents,nonlinear_prelureg_recons,
                        nonlinear_matreg_latents,nonlinear_matreg_recons,true_labels,
                        show=False,save_fn=''):

    fig_layout=\
    [['Original "Latents"', 'Original "Latents"','Linear Decoder Latents', 'Linear Decoder Latents','Deep Linear Decoder Latents', 'Deep Linear Decoder Latents','Deep unconstrained decoder Latents','Deep unconstrained decoder Latents','Nonlinear regularized decoder Latents (prelu)','Nonlinear regularized decoder Latents (prelu)','Nonlinear regularized decoder Latents (prelu)','Nonlinear regularized decoder Latents (prelu)'],\
    ['Original data','Original data','Linear Decoder Reconstructions', 'Linear Decoder Reconstructions','Deep Linear Decoder Reconstructions', 'Deep Linear Decoder Reconstructions','Deep unconstrained decoder Reconstructions','Deep unconstrained decoder Reconstructions','nonlinear regularized decoder Reconstructions (prelu)','nonlinear regularized decoder Reconstructions (prelu)','nonlinear regularized decoder Reconstructions (mat)','nonlinear regularized decoder Reconstructions (mat)']]

    #plt.close('all')
    fig,axs = plt.subplot_mosaic(fig_layout,figsize=(25,5))
    labels = np.unique(true_labels)

    if train_data.shape[1] > 2:
        train_data = PCA(n_components=2).fit_transform(train_data)
        linear_recons = PCA(n_components=2).fit_transform(linear_recons)
        deep_linear_recons = PCA(n_components=2).fit_transform(deep_linear_recons)
        nonlinear_recons = PCA(n_components=2).fit_transform(nonlinear_recons)
    for label in labels:
        data_inds = true_labels == label
        axs['Original "Latents"'].scatter(true_latents[data_inds,0],true_latents[data_inds,1],label='Generating data')
        axs['Original data'].scatter(train_data[data_inds,0],train_data[data_inds,1],label='Training data')

        axs['Linear Decoder Latents'].scatter(linear_latents[data_inds,0],linear_latents[data_inds,1],label='Linear embedding')
        axs['Linear Decoder Reconstructions'].scatter(linear_recons[data_inds,0],linear_recons[data_inds,1],label='Linear reconstruction')

        axs['Deep Linear Decoder Latents'].scatter(deep_linear_latents[data_inds,0],deep_linear_latents[data_inds,1],label='Deep linear embedding')
        axs['Deep Linear Decoder Reconstructions'].scatter(deep_linear_recons[data_inds,0],deep_linear_recons[data_inds,1],label='Deep linear reconstruction')

        axs['Deep unconstrained decoder Latents'].scatter(nonlinear_latents[data_inds,0],nonlinear_latents[data_inds,1],label='Deep nonlinear embedding')
        axs['Deep unconstrained decoder Reconstructions'].scatter(nonlinear_recons[data_inds,0],nonlinear_recons[data_inds,1],label='Deep nonlinear reconstruction')

        axs['Nonlinear regularized decoder Latents (prelu)'].scatter(nonlinear_prelureg_latents[data_inds,0],nonlinear_prelureg_latents[data_inds,1],label='Deep prelu reg nonlinear embedding')
        axs['Nonlinear regularized decoder Reconstructions (prelu)'].scatter(nonlinear_prelureg_recons[data_inds,0],nonlinear_prelureg_recons[data_inds,1],label='Deep prelu reg nonlinear reconstruction')

        axs['nonlinear regularized decoder Latents (mat)'].scatter(nonlinear_matreg_latents[data_inds,0],nonlinear_matreg_latents[data_inds,1],label='Deep matreg nonlinear embedding')
        axs['nonlinear regularized decoder Reconstructions (mat)'].scatter(nonlinear_matreg_recons[data_inds,0],nonlinear_matreg_recons[data_inds,1],label='Deep matreg nonlinear reconstruction')

    for key in axs.keys():
        ax = axs[key]
        ax = format_axis(ax,xlabel='dim 1',ylabel='dim 2',title=key,\
                         xticks=ax.get_xticks(),
                         yticks=ax.get_yticks(),
                         xlims=ax.get_xlim(),
                         ylims=ax.get_ylim())

    plt.tight_layout()
    if show:
        plt.show()
    elif save_fn != '':
        plt.savefig(save_fn)
    plt.close()

    