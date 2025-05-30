import numpy as np
import matplotlib.pyplot as plt


def make_toy_plot(true_latents,train_data,\
                  linear_latents,linear_recons,\
                    deep_linear_latents,deep_linear_recons,\
                        nonlinear_latents,nonlinear_recons,true_labels,
                        show=False,save_fn=''):

    fig_layout=\
    [['Original "Latents"', 'Original "Latents"','Linear Decoder Latents', 'Linear Decoder Latents','Deep Linear Decoder Latents', 'Deep Linear Decoder Latents','Deep unconstrained decoder Latents','Deep unconstrained decoder Latents'],\
    ['Original data','Original data','Linear Decoder Reconstructions', 'Linear Decoder Reconstructions','Deep Linear Decoder Reconstructions', 'Deep Linear Decoder Reconstructions','Deep unconstrained decoder Reconstructions','Deep unconstrained decoder Reconstructions']]

    plt.close('all')
    fig = plt.Figure(figsize=(10,5))
    axs = fig.subplot_mosaic(fig_layout)
    labels = np.unique(true_labels)
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

    plt.tight_layout()

    