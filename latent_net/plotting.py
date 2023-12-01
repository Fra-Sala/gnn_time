from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from latent_net import preprocessing_scale  
import matplotlib.colors as mcolors
import torch

def plot_fields(SNAP, results, scaler_all, HyperParams, dataset, position_dataset, PARAMS, TIMES):
    """
    Plots the field solution for a given snapshot, the ground truth, and the error field.

    The function takes in the following inputs:

    SNAP: integer value indicating the snapshot to be plotted.
    results: array of shape (num_samples, num_features), representing the network's output.
    scaler_all: instance of the scaler used to scale the data.
    HyperParams: instance of the Autoencoder parameters class containing information about the network architecture and training.
    dataset: array of shape (num_samples, 3), representing the triangulation of the spatial domain.
    xyz: list of arrays of shape (num_samples, num_features), containing the x, y and z-coordinates of the domain.
    params: array of shape (num_features,), containing the parameters associated with each snapshot.
    The function generates a plot of the field solution and saves it to disk using the filepath specified in HyperParams.net_dir.
    """

    fig = plt.figure(figsize=(15, 5))
    z_net = preprocessing_scale.inverse_normalize_input(results, scaler_all, SNAP)
    z_net = z_net.squeeze(0).squeeze(-1)
    xx = dataset.xx
    yy = dataset.yy
    rel_error_field = abs(dataset.U[:, SNAP] - z_net) / np.linalg.norm(dataset.U[:, SNAP], 2)

    triang = np.asarray(dataset.T - 1)
    cmap = cm.get_cmap(name='jet', lut=None)
    norm1 = mcolors.Normalize(vmin=z_net.min(), vmax=z_net.max())
    gs1 = gridspec.GridSpec(1, 3)  # Change to 2 columns for 2 subplots

    # Subplot 1
    ax1 = plt.subplot(gs1[0, 0])
    cs1 = ax1.tricontourf(xx[:, SNAP], yy[:, SNAP], triang, z_net, 100, cmap=cmap, norm=norm1)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cbar1 = plt.colorbar(cs1, cax=cax1)
    cbar1.formatter.set_powerlimits((0, 0))
    cbar1.update_ticks()
    ax1.set_aspect('equal', 'box')
    ax1.set_title('Prediction for $\mu$ = ' + str(np.around(PARAMS[SNAP], 2)) + str(' at t = ') + str(
        np.around(TIMES[SNAP], 2)))

    # Subplot 2
    #average_dataset = torch.mean(dataset.U[:, :90], dim=1)
    #norm2= mcolors.Normalize(vmin=average_dataset.min(), vmax=average_dataset.max())
    norm2 = mcolors.Normalize(vmin=dataset.U[:, SNAP].min(), vmax=dataset.U[:, SNAP].max())
    ax2 = plt.subplot(gs1[0, 1])  # Add second subplot
    cs2 = ax2.tricontourf(xx[:, SNAP], yy[:, SNAP], triang, dataset.U[:,SNAP], 100, cmap=cmap, norm=norm2)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cbar2 = plt.colorbar(cs2, cax=cax2)
    cbar2.formatter.set_powerlimits((0, 0))
    cbar2.update_ticks()
    ax2.set_aspect('equal', 'box')
    ax2.set_title('Ground truth $\mu$ = ' + str(np.around(PARAMS[SNAP], 2)) + str(' at t = ') + str(
        np.around(TIMES[SNAP], 2)))

    # Subplot 3
    norm3 = mcolors.Normalize(vmin=rel_error_field.min(), vmax=rel_error_field.max())
    ax3 = plt.subplot(gs1[0, 2])
    cs3 = ax3.tricontourf(xx[:, SNAP], yy[:, SNAP], triang, rel_error_field, 100, cmap=cmap, norm=norm3)
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.1)
    cbar3 = plt.colorbar(cs3, cax=cax3)
    cbar3.formatter.set_powerlimits((0, 0))
    cbar3.update_ticks()
    ax3.set_aspect('equal', 'box')
    ax3.set_title('Relative Error')

    # Adjust layout
    plt.tight_layout()
    plt.savefig(HyperParams.net_dir + 'field_solution_' + str(SNAP) + '.png', bbox_inches='tight', dpi=500)
    plt.show()



def plot_latent(stn_evolution, time_evolution):
    
    plt.figure()
    for i in range(len(stn_evolution[0])):
        stn_i = []
        for j in range(len(stn_evolution)):
            stn_i.append(stn_evolution[j][i].item())
        plt.plot(time_evolution, stn_i)
        plt.xlabel('Time')
        plt.ylabel('Latent state')
        plt.title('Latent state evolution')
    plt.show()