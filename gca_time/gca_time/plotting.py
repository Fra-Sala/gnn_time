import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mcolors
from gca_time import preprocessing
from matplotlib.ticker import FuncFormatter
from scipy import interpolate
import matplotlib.tri as mtri



params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

def plot_loss(HyperParams):
    """
    Plots the history of losses during the training of the latent net + decoder.

    Parameters:
    HyperParams (object): An object containing the parameters of the architecture.
    """

    history = np.load(HyperParams.net_dir+'history'+HyperParams.net_run+'.npy', allow_pickle=True).item()
    history_test = np.load(HyperParams.net_dir+'history_test'+HyperParams.net_run+'.npy', allow_pickle=True).item()
    ax = plt.figure().gca()
    ax.semilogy(history['train'])
    ax.semilogy(history_test['test'], '--')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.title('Loss over training epochs')
    plt.legend(['loss_train','loss_test'])
    plt.savefig(HyperParams.net_dir+'history_losses'+HyperParams.net_run+'.png', bbox_inches='tight', dpi=500)


def plot_fields(SNAP, results, scaler_all, HyperParams, dataset, PARAMS, TIMES):
    """
    Plots the field solution for a given snapshot, the ground truth, and the error field.

    The function takes in the following inputs:

    SNAP: integer value indicating the snapshot to be plotted.
    results: array of shape (num_samples, num_features), representing the network's output.
    scaler_all: numpy.ndarray of scaling variables.
    HyperParams: instance of the Autoencoder parameters class containing information about the network architecture and training.
    dataset: array of shape (num_samples, 3), representing the Fenics dataset.
    PARAMS: array of shape (num_snap,), containing the parameters associated with each snapshot.
    TIMES: array of shape (num_snap,), containing the time associated with each snapshot.
    
    The function generates a plot of the field solution and saves it to disk using the filepath specified in HyperParams.net_dir.
    """

    fig = plt.figure(figsize=(15, 5))
    z_net = preprocessing.inverse_normalize_input(results[SNAP, :, :], scaler_all, SNAP, HyperParams)
    # Now, if the second dimension of z_net is == 2, change z_net into a 1D array by computing the norm of each row
    # set required grad false to z_net
    z_net = z_net.detach().numpy()
    
    if HyperParams.dim_sol == 1:
        z_net = z_net[:, 0]
        ground_truth = dataset.U[:, SNAP]
    if HyperParams.dim_sol == 2:
        z_net = np.linalg.norm(z_net, axis=1)
        ground_truth = np.linalg.norm(np.column_stack((dataset.VX[:, SNAP], dataset.VY[:, SNAP])), axis=1)
    
    xx = dataset.xx
    yy = dataset.yy
    rel_error_field = abs(ground_truth- z_net) / np.linalg.norm(ground_truth, 2)

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
    norm2 = mcolors.Normalize(vmin=ground_truth.min(), vmax=ground_truth.max())
    ax2 = plt.subplot(gs1[0, 1])  # Add second subplot
    cs2 = ax2.tricontourf(xx[:, SNAP], yy[:, SNAP], triang, ground_truth, 100, cmap=cmap, norm=norm2)
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
    plt.savefig(HyperParams.net_dir + 'field_solution_SNAP' + str(SNAP) + '.png', bbox_inches='tight', dpi=500)
    plt.show()




def plot_latent(SNAP, latents, params, time_evolution, HyperParams):
    """
    This function plots the evolution of latent states over time and saves the plot as a .png file.

    Parameters:
    SNAP (int): The snapshot number.
    latents (np.ndarray): The latent states.
    params (list): The parameters.
    time_evolution (np.ndarray): The time evolution.
    HyperParams (object): The hyperparameters.

    Returns:
    None
    """

    # Create a new figure
    plt.figure(figsize=(6, 5))

    # Convert tensors to numpy arrays
    latents = latents.detach().numpy()
    time_evolution = time_evolution.detach().numpy()

    # Calculate length of a simulation and sim index
    sequence_length = latents.shape[0] // len(params)
    sequence_number = int(SNAP / latents.shape[0] *sequence_length)
    start = sequence_number * sequence_length
    end = start + sequence_length

    # Plot each latent state over time
    for i in range(HyperParams.bottleneck_dim):
        stn_evolution = latents[start:end, i]
        times = latents[start:end, -1]
        plt.plot(times, stn_evolution)

    plt.xlabel('$t$')
    plt.ylabel('$s(t)$')
    plt.title('Latent state evolution $\mu = $'+ str(np.around(params[sequence_number],2)))
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(f"{HyperParams.net_dir}_latents_evolution_{HyperParams.net_run}_SNAP_{SNAP}.png", bbox_inches='tight', dpi=500)
    plt.show()

def plot_time_extrapolation(results, scaler_all, HyperParams, dataset, params, times, n_train_instants):
    """
    This function plots a metric of the error when performing a time extrapolation of the 
    predictions.

    Parameters:
    results (numpy.ndarray): The tensor of the predicted fields (entire dataset, training and testing).
    scaler_all (numpy.ndarray): The scaler used for normalization.
    HyperParams (object): The hyperparameter object used in the model.
    dataset (object): The dataset used in the model.
    params (list): The parameters used in the model that define the training simulations 
                   (the extrapolation concerns only the time, not the parameter space).
    times (list): The times at which the results are obtained.
    n_train_instants (int): The number of training instants per simulation (the remaining instants are used for
                    the extrapolation).

    Returns:
    None
    """

    plt.figure(figsize=(8, 5))

    # Exclude t = 0
    times = times[1:]

    # Initialize the Z_net and ground_truth arrays
    Z_net = np.zeros((dataset.VX.shape[1], dataset.VX.shape[0],2))
    ground_truth = np.zeros((dataset.VX.shape[1], dataset.VX.shape[0],2))

    # Scale back the results over the entire dataset
    for i in range(results.shape[0]):
        out = preprocessing.inverse_normalize_input(results[i, :, :], scaler_all, i, HyperParams)
        out = out.detach().numpy()
        Z_net[i, :, :] = out
        ground_truth[i, :, :] = np.column_stack((dataset.VX[:, i], dataset.VY[:, i]))

    # Calculate the norms of the ground truth
    ground_truth_norms = np.linalg.norm(ground_truth, axis=2)
    ground_truth_norms_mean = np.mean(ground_truth_norms, axis=1)

    # Calculate the NRMSE for each parameter and plot it
    for (j, param) in enumerate(params):
        NRMSE_list = []
        for i in range(len(times)):
            pred = Z_net[i+j*len(times)]
            target = ground_truth[i+j*len(times)]
            non_zero_indices = np.where(np.linalg.norm(target, axis=1) != 0)
            err = np.sum((np.linalg.norm(pred[non_zero_indices] - target[non_zero_indices], axis=1)**2) / (ground_truth_norms_mean[i+j*len(times)]**2))
            err = err / len(non_zero_indices[0])
            NRMSE = np.sqrt(err)
            NRMSE_list.append(NRMSE)
        plt.semilogy(times, np.array(NRMSE_list), label = '$\mu$ = ' + str(np.around(param,2)))
        plt.semilogy(times[:n_train_instants], np.array(NRMSE_list)[:n_train_instants], 'o', color='blue')
        plt.semilogy(times[n_train_instants:], np.array(NRMSE_list)[n_train_instants:], 'x', color='red')
        plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xlabel('$t$')
    plt.ylabel('NRMSE')
    plt.legend(loc='best')
    plt.title('NRMSE for time extrapolation')
    plt.savefig(HyperParams.net_dir+'NRMSE_time_extrapolation'+HyperParams.net_run+'.png', bbox_inches='tight', dpi=500)
    plt.show()



def plot_error(results, dataset, scaler_all, HyperParams, params, PARAMS, time, TIMES, train_trajectories):
    """
    This function plots the relative error between the predicted and actual results, for a number of parameters larger than 1
    using time and each of the other n_params-1 parameters for the plot.

    Parameters:
    res (ndarray): The predicted results
    VAR_all (ndarray): The actual results
    scaler_all (object): The scaler object used for scaling the results
    HyperParams (object): The HyperParams object holding the necessary hyperparameters
    mu1_range (ndarray): Range of the first input variable
    mu2_range (ndarray): Range of the second input variable
    params (ndarray): The input variables
    train_trajectories (ndarray): The indices of the training data
    vars (str): The name of the variable being plotted
    """
    vars = 'vs $\mu$ and $t$'
    Z_net = np.zeros((dataset.VX.shape[1], dataset.VX.shape[0],2))
    ground_truth = np.zeros((dataset.VX.shape[1], dataset.VX.shape[0],2))

    # Scale back the results over the entire dataset
    for i in range(results.shape[0]):
        out = preprocessing.inverse_normalize_input(results[i, :, :], scaler_all, i, HyperParams)
        out = out.detach().numpy()
        Z_net[i, :, :] = out
        ground_truth[i, :, :] = np.column_stack((dataset.VX[:, i], dataset.VY[:, i]))
    # Take only the magnitude of velocity
    Z_net = np.linalg.norm(Z_net, axis=2)
    ground_truth = np.linalg.norm(ground_truth, axis=2)
    # Calculate the relative error
    error = np.linalg.norm(Z_net - ground_truth, axis=1) / np.mean(np.linalg.norm(ground_truth, axis=1))
        
    colors = 0.0
    area = 0.0 
    tr_pt_1 = PARAMS[train_trajectories]
    tr_pt_2 = TIMES[train_trajectories]

    X1, X2 = np.meshgrid(params, time, indexing='ij')
    output = np.reshape(error, (len(params), len(time)))
    fig = plt.figure('Relative Error '+vars)
    ax = fig.add_subplot()
    colors = output.flatten()
    area = output.flatten()*500

    sc = plt.scatter(X1.flatten(), X2.flatten(), s=area, c= colors, alpha=0.5, cmap=cm.coolwarm)
    plt.colorbar(sc, format=FuncFormatter(lambda x, pos: f'{x:.1e}'))
    ax.set(xlim = [-10,10], #xlim=tuple([np.min(mu_i_range), np.max(mu_i_range)]
            ylim=[0,2],
            xlabel=f'$\mu$',
            ylabel=f'$t$')
        
    ax.plot(tr_pt_1, tr_pt_2,'*r')
    ax.set_title('Relative Error '+vars)

    plt.tight_layout()
    plt.savefig(HyperParams.net_dir+'relative_error_scatter'+HyperParams.net_run+'_'+'.png', transparent=True, dpi=500)
    plt.show()





# def plot_latent(HyperParams, latents, latents_estimation):
#     """
#     Plot the original and estimated latent spaces
    
#     Parameters:
#     HyperParams (obj): object containing the Autoencoder parameters 
#     latents (tensor): tensor of original latent spaces
#     latents_estimation (tensor): tensor of estimated latent spaces
#     """

#     plt.figure()
#     for i1 in range(HyperParams.bottleneck_dim):
#         plt.plot(latents[:,i1].detach(), '--')
#         plt.plot(latents_estimation[:,i1].detach(),'-')
#     plt.savefig(HyperParams.net_dir+'latents'+HyperParams.net_run+'.png', bbox_inches='tight')
    
#     green_diamond = dict(markerfacecolor='g', marker='D')
#     _, ax = plt.subplots()
#     ax.boxplot(latents_estimation.detach().numpy(), flierprops=green_diamond)
#     plt.savefig(HyperParams.net_dir+'box_plot_latents'+HyperParams.net_run+'.png', bbox_inches='tight', dpi=500)
    
# def plot_error_multip(res, VAR_all, scaler_all, HyperParams, mu_space, params, train_trajectories, vars, p1=0, p2=-1):
#     """
#     This function plots the relative error between the predicted and actual results, for a number of parameters larger than 1
#     using time and each of the other n_params-1 parameters for the plot.

#     Parameters:
#     res (ndarray): The predicted results
#     VAR_all (ndarray): The actual results
#     scaler_all (object): The scaler object used for scaling the results
#     HyperParams (object): The HyperParams object holding the necessary hyperparameters
#     mu1_range (ndarray): Range of the first input variable
#     mu2_range (ndarray): Range of the second input variable
#     params (ndarray): The input variables
#     train_trajectories (ndarray): The indices of the training data
#     vars (str): The name of the variable being plotted
#     """

#     u_hf = scaling.inverse_scaling(VAR_all, scaler_all, HyperParams.scaling_type)
#     u_app = scaling.inverse_scaling(res, scaler_all, HyperParams.scaling_type)
#     error = np.linalg.norm(u_app - u_hf, axis=0) / np.mean(np.linalg.norm(u_hf, axis=0))
#     #ipdb.set_trace()
#     # mu_space_cp = mu_space
#     # #Create a ndarray containing the parameters
#     # time = mu_space_cp.pop()
#     # params = []
#     # for i in range(len(mu_space_cp[0])):
#     #     set_coeff = [arr[i] for arr in mu_space_cp]
#     #     for j in range(len(time)):
#     #         new_set = np.concatenate((set_coeff, [time[j]]), axis = 0)
#     #         params.append(new_set)
#     # params = np.array(params)
    
#     # For each parameter, realize a plot
#     times = mu_space[-1]
#     n_params = params.shape[1]
#     colors = 0.0
#     area = 0.0
#     for i in range(n_params-1):  
#         mu_i_range = mu_space[i]
     
#         tr_pt_1 = params[train_trajectories, i]
#         tr_pt_2 = params[train_trajectories, -1]
#         # if n_params > 2:
#         #     rows, ind = np.unique(params[:, [p1, p2]], axis=0, return_inverse=True)
#         #     print(rows, ind)
#         #     indices_dict = defaultdict(list)
#         #     ipdb.set_trace()
#         #     [indices_dict[tuple(rows[i])].append(idx) for idx, i in enumerate(ind)]
#         #     error = np.array([np.mean(error[indices]) for indices in indices_dict.values()])
#         #     tr_pt = [i for i in indices_dict if any(idx in train_trajectories for idx in indices_dict[i])]
#         #     tr_pt_1 = [t[0] for t in tr_pt]
#         #     tr_pt_2 = [t[1] for t in tr_pt]
#         X1, X2 = np.meshgrid(mu_i_range, times, indexing='ij')
#         output = np.reshape(error, (len(mu_i_range), len(times)))
#         fig = plt.figure('Relative Error '+vars)
#         ax = fig.add_subplot()
#         # z_anchor = np.zeros_like(output)
#         # dx= 0.3
#         # dy = 0.02
#         # ax.bar3d(X1.flatten(), X2.flatten(), z_anchor.flatten(), dx, dy, output.flatten(), cmap=cm.coolwarm,  alpha = 1)
#         if i == 0:
#             colors = output.flatten()
#             area = output.flatten()*1000

#         sc = plt.scatter(X1.flatten(), X2.flatten(), s=area, c= colors, alpha=0.5, cmap=cm.coolwarm)
#         plt.colorbar(sc)
#         #ax.scatter(X1, X2, output, cmap=cm.coolwarm, color='blue')
#         #ax.contour(X1, X2, output, zdir='z', offset=output.min(), cmap=cm.coolwarm)
#         ax.set(xlim = [-10,10], #xlim=tuple([np.min(mu_i_range), np.max(mu_i_range)]
#                ylim=[0,2],
#                xlabel=f'$\mu_{str(i+1)}$',
#                ylabel=f'$t$')
#                #zlabel='$\\epsilon_{GCA}(\\mathbf{\mu})$')
#         ax.plot(tr_pt_1, tr_pt_2,'*r')
#         # plt.ticklabel_format(axis='z', style='sci', scilimits=(0, 0))
#         ax.set_title('Relative Error '+vars)
#         # ax.zaxis.offsetText.set_visible(False)
#         # exponent_axis = np.floor(np.log10(max(ax.get_zticks()))).astype(int)
#         # ax.text2D(0.9, 0.82, "1e"+str(exponent_axis), transform=ax.transAxes, fontsize="x-large")
#         plt.tight_layout()
#         plt.savefig(HyperParams.net_dir+'relative_error_'+vars+HyperParams.net_run+'_'+str(i+1)+'.png', transparent=True, dpi=500)
#         plt.show()
    

# def plot_error(res, VAR_all, scaler_all, HyperParams, mu_space, params, train_trajectories, vars, p1=0, p2=-1):
#     """
#     This function plots the relative error between the predicted and actual results.

#     Parameters:
#     res (ndarray): The predicted results
#     VAR_all (ndarray): The actual results
#     scaler_all (object): The scaler object used for scaling the results
#     HyperParams (object): The HyperParams object holding the necessary hyperparameters
#     mu1_range (ndarray): Range of the first input variable
#     mu2_range (ndarray): Range of the second input variable
#     params (ndarray): The input variables
#     train_trajectories (ndarray): The indices of the training data
#     vars (str): The name of the variable being plotted
#     """

#     u_hf = scaling.inverse_scaling(VAR_all, scaler_all, HyperParams.scaling_type)
#     u_app = scaling.inverse_scaling(res, scaler_all, HyperParams.scaling_type)
#     error = np.linalg.norm(u_app - u_hf, axis=0) / np.mean(np.linalg.norm(u_hf, axis=0))
#     mu1_range = mu_space[p1]
#     mu2_range = mu_space[p2]
#     n_params = params.shape[1]
#     tr_pt_1 = params[train_trajectories, p1]
#     tr_pt_2 = params[train_trajectories, p2]
#     if n_params > 2:
#         rows, ind = np.unique(params[:, [p1, p2]], axis=0, return_inverse=True)
#         print(rows, ind)
#         indices_dict = defaultdict(list)
#         [indices_dict[tuple(rows[i])].append(idx) for idx, i in enumerate(ind)]
#         error = np.array([np.mean(error[indices]) for indices in indices_dict.values()])
#         tr_pt = [i for i in indices_dict if any(idx in train_trajectories for idx in indices_dict[i])]
#         tr_pt_1 = [t[0] for t in tr_pt]
#         tr_pt_2 = [t[1] for t in tr_pt]
#     X1, X2 = np.meshgrid(mu1_range, mu2_range, indexing='ij')
#     output = np.reshape(error, (len(mu1_range), len(mu2_range)))
#     fig = plt.figure('Relative Error '+vars)
#     ax = fig.add_subplot(projection='3d')
#     ax.plot_surface(X1, X2, output, cmap=cm.coolwarm, color='blue')
#     ax.contour(X1, X2, output, zdir='z', offset=output.min(), cmap=cm.coolwarm)
#     ax.set(xlim=tuple([mu1_range[0], mu1_range[-1]]),
#            ylim=tuple([mu2_range[0], mu2_range[-1]]),
#            xlabel=f'$\mu_{str((p1%n_params)+1)}$',
#            ylabel=f'$\mu_{str((p2%n_params)+1)}$',
#            zlabel='$\\epsilon_{GCA}(\\mathbf{\mu})$')
#     ax.plot(tr_pt_1, tr_pt_2, output.min()*np.ones(len(tr_pt_1)), '*r')
#     plt.ticklabel_format(axis='z', style='sci', scilimits=(0, 0))
#     ax.set_title('Relative Error '+vars)
#     ax.zaxis.offsetText.set_visible(False)
#     exponent_axis = np.floor(np.log10(max(ax.get_zticks()))).astype(int)
#     ax.text2D(0.9, 0.82, "1e"+str(exponent_axis), transform=ax.transAxes, fontsize="x-large")
#     plt.tight_layout()
#     plt.savefig(HyperParams.net_dir+'relative_error_'+vars+HyperParams.net_run+'.png', transparent=True, dpi=500)


# def plot_fields(SNAP, results, scaler_all, HyperParams, dataset, xyz, params):
#     """
#     Plots the field solution for a given snapshot.

#     The function takes in the following inputs:

#     SNAP: integer value indicating the snapshot to be plotted.
#     results: array of shape (num_samples, num_features), representing the network's output.
#     scaler_all: instance of the scaler used to scale the data.
#     HyperParams: instance of the Autoencoder parameters class containing information about the network architecture and training.
#     dataset: array of shape (num_samples, 3), representing the triangulation of the spatial domain.
#     xyz: list of arrays of shape (num_samples, num_features), containing the x, y and z-coordinates of the domain.
#     params: array of shape (num_features,), containing the parameters associated with each snapshot.
#     The function generates a plot of the field solution and saves it to disk using the filepath specified in HyperParams.net_dir.
#     """

#     fig = plt.figure()    
#     Z_net = scaling.inverse_scaling(results, scaler_all, HyperParams.scaling_type)
#     z_net = Z_net[:, SNAP]
#     xx = xyz[0]
#     yy = xyz[1]
#     if dataset.dim == 2:
#         triang = np.asarray(dataset.T - 1)
#         cmap = cm.get_cmap(name='jet', lut=None)
#         gs1 = gridspec.GridSpec(1, 1)
#         ax = plt.subplot(gs1[0, 0])
#         cs = ax.tricontourf(xx[:, SNAP], yy[:, SNAP], triang, z_net, 100, cmap=cmap)
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.1)
#         cbar = plt.colorbar(cs, cax=cax)
#     elif dataset.dim == 3:
#         zz = xyz[2]
#         ax = fig.add_subplot(projection='3d')
#         cax = inset_axes(ax, width="5%", height="60%", loc="center left", 
#                          bbox_to_anchor=(1.15, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
#         cmap = cm.get_cmap(name='jet', lut=None) 
#         p = ax.scatter(xx[:, SNAP], yy[:, SNAP], zz[:, SNAP], c=z_net, cmap=cmap, linewidth=0.5)
#         cbar = fig.colorbar(p, ax=ax, cax=cax)
#         ax.set_xlabel('$x$')
#         ax.set_ylabel('$y$')
#         ax.set_zlabel('$z$')
#         ax.locator_params(axis='both', nbins=5)
#     tick_locator = MaxNLocator(nbins=5)
#     cbar.locator = tick_locator
#     cbar.formatter.set_powerlimits((0, 0))
#     cbar.update_ticks()
#     plt.tight_layout()
#     ax.set_aspect('equal', 'box')
#     ax.set_title('Solution field for $\mu$ = '+str(np.around(params[SNAP].detach().numpy(), 2)))
#     plt.savefig(HyperParams.net_dir+'field_solution_'+str(SNAP)+''+HyperParams.net_run+'.png', bbox_inches='tight', dpi=500)


# def plot_error_fields(SNAP, results, VAR_all, scaler_all, HyperParams, dataset, xyz, params):
#     """
#     This function plots a contour map of the error field of a given solution of a scalar field.
#     The error is computed as the absolute difference between the true solution and the predicted solution,
#     normalized by the 2-norm of the true solution.

#     Inputs:
#     SNAP: int, snapshot of the solution to be plotted
#     results: np.array, predicted solution
#     VAR_all: np.array, true solution
#     scaler_all: np.array, scaling information used in the prediction
#     HyperParams: class, model architecture and training parameters
#     dataset: np.array, mesh information
#     xyz: list of arrays of shape (num_samples, num_features), containing the x, y and z-coordinates of the domain.
#     params: np.array, model parameters
#     """

#     fig = plt.figure()
#     Z = scaling.inverse_scaling(VAR_all, scaler_all, HyperParams.scaling_type)
#     Z_net = scaling.inverse_scaling(results, scaler_all, HyperParams.scaling_type)
#     z = Z[:, SNAP]
#     z_net = Z_net[:, SNAP]
#     error = abs(z - z_net)/np.linalg.norm(z, 2)
#     xx = xyz[0]
#     yy = xyz[1]
#     if dataset.dim == 2:
#         triang = np.asarray(dataset.T - 1)
#         cmap = cm.get_cmap(name='jet', lut=None) 
#         gs1 = gridspec.GridSpec(1, 1)
#         ax = plt.subplot(gs1[0, 0])   
#         cs = ax.tricontourf(xx[:, SNAP], yy[:, SNAP], triang, error, 100, cmap=cmap)
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.1)
#         cbar = plt.colorbar(cs, cax=cax)
#     elif dataset.dim == 3:
#         zz = xyz[2]
#         ax = fig.add_subplot(projection='3d')
#         cax = inset_axes(ax, width="5%", height="60%", loc="center left", 
#                          bbox_to_anchor=(1.15, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
#         cmap = cm.get_cmap(name='jet', lut=None) 
#         p = ax.scatter(xx[:, SNAP], yy[:, SNAP], zz[:, SNAP], c=error, cmap=cmap, linewidth=0.5)
#         cbar = fig.colorbar(p, ax=ax, cax=cax)
#         ax.set_xlabel('$x$')
#         ax.set_ylabel('$y$')
#         ax.set_zlabel('$z$')
#         ax.locator_params(axis='both', nbins=5)
#     tick_locator = MaxNLocator(nbins=5)
#     cbar.locator = tick_locator
#     cbar.formatter.set_powerlimits((0, 0))
#     cbar.update_ticks()
#     plt.tight_layout()
#     ax.set_aspect('equal', 'box')
#     ax.set_title('Error field for $\mu$ = '+str(np.around(params[SNAP].detach().numpy(), 2)))
#     plt.savefig(HyperParams.net_dir+'error_field_'+str(SNAP)+''+HyperParams.net_run+'.png', bbox_inches='tight', dpi=500)
