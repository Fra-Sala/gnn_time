import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
# from latent_net import shuffle_dataset, rff_fun
# import rff
import os
from lid_driven_cavity_fenics import gaussian_process
import matplotlib.pyplot as plt

def test(dyn_model, rec_model, device, param, t, position_dataset, num_necessary_batches, HyperParams):
    """
    This function evaluates the dynamic and reconstruction models.

    Parameters:
    dyn_model (torch.nn.Module): The dynamic model to be tested.
    rec_model (torch.nn.Module): The reconstruction model to be tested.
    device (torch.device): The device on which the computations will be performed.
    param (float): The parameter for the Gaussian process.
    t (float): The time at which the models are tested.
    queried_positions (torch.Tensor): The positions at which the models are tested.
    HyperParams (object): An object containing various hyperparameters.

    Returns:
    velocity_pred (torch.Tensor): The predicted velocity by the reconstruction model.
    stn_vec (list): A list of the state vectors at each time step.
    t_integration (list): A list of the time steps at which the models are tested.
    """

    dyn_model.eval()
    rec_model.eval()
    stn = torch.zeros(HyperParams.dim_latent, device=device)
    Z_net = torch.zeros((len(position_dataset), HyperParams.dim))

    with torch.no_grad():
        
        stn_vec = []
        t_integration = []
        t_integration.append(np.zeros((1,)))
        stn = torch.zeros(HyperParams.dim_latent, device=device)
        stn_vec.append(stn)
        num_integrations = round(float((t))/HyperParams.dt)

        for j in range(num_integrations):
            u_t_needed = gaussian_process.eval_u_t(t_integration[-1], param, HyperParams.T_f)
            u_t_needed = torch.tensor(u_t_needed, device=device)
            dyn_input = torch.cat((u_t_needed, stn), dim=0)
            stn_derivative = dyn_model(dyn_input)
            stn = stn + HyperParams.dt * stn_derivative
            stn_vec.append(stn)
            t_integration.append(t_integration[-1] + HyperParams.dt)
        # Loop over the positions to have a prediction for the entire domain

        # Take a batch of positions
        for j in range(len(position_dataset)):
            x, y = position_dataset[j,0], position_dataset[j,1]
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            rec_input = torch.cat((stn, x.to(device), y.to(device)), dim=0)
            velocity_pred = rec_model(rec_input)
            Z_net[j, :] = velocity_pred



        # for i in range(num_necessary_batches):
        #     queried_positions = position_dataset[i*HyperParams.batch_pos_size:(i+1)*HyperParams.batch_pos_size]
        #     x_pos, y_pos = queried_positions[:,0], queried_positions[:,1]
        #     counter = 0
        #     for x, y in zip(x_pos, y_pos):
        #         x = x.unsqueeze(0)
        #         y = y.unsqueeze(0)
        #         rec_input = torch.cat((stn, x.to(device), y.to(device)), dim=0)
        #         velocity_pred = rec_model(rec_input)
        #         if HyperParams.dim == 1:
        #             Z_net[i*HyperParams.batch_pos_size:i*HyperParams.batch_pos_size+counter, 0] = velocity_pred
        #         if HyperParams.dim == 2:
        #             Z_net[i*HyperParams.batch_pos_size:i*HyperParams.batch_pos_size+counter, 0] = velocity_pred[0]
        #             Z_net[i*HyperParams.batch_pos_size:i*HyperParams.batch_pos_size+counter, 1] = velocity_pred[1]
                        
        #         counter +=1

                        
         
          

    return Z_net, stn_vec, t_integration