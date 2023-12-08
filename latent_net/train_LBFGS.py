import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
# from latent_net import shuffle_dataset, rff_fun
# import rff
import os
from lid_driven_cavity_fenics import gaussian_process
import matplotlib.pyplot as plt

# Set default data type to float32
#torch.set_default_dtype(torch.float32)

# Function to train dynamic and rec networks
def train_dyn_rec_nets(dyn_model, rec_model, optimizer, device,\
                        params_train, params_test, times, train_loader, test_loader, position_dataset, train_snapshots, test_snapshots, HyperParams):
    
    train_history = {"loss": [], "l1": [], "l2": []}
    test_history = {"loss": [], "l1": [], "l2": []}
    min_test_loss = np.Inf
    dyn_model.to(device)
    rec_model.to(device)
    # Progress bar
    loop = tqdm(range(HyperParams.max_epochs))
    # Take the train data
    data_train_iterator = iter(train_loader)
    data_train = next(data_train_iterator).to(device)
    # Take the test data
    data_test_iterator = iter(test_loader)
    data_test = next(data_test_iterator).to(device)
    # Check how many batches of positions can be created
    position_dataset.to(device)
    
    
    for epoch in loop:
        #pos_batch_sampler, position_loader = shuffle_dataset.shuffle_position_dataset(position_dataset, HyperParams, seed=epoch)
        train_loss = train_one_epoch(dyn_model, rec_model, optimizer,  device, params_train,\
                                      times, data_train, position_dataset, train_snapshots, HyperParams)

        train_history["loss"].append(train_loss)          
        test_loss = evaluate_model(dyn_model, rec_model, device, params_test, \
                                   times, data_test, position_dataset, test_snapshots, HyperParams)
    
        test_history["loss"].append(test_loss)
     
        # Update the progress bar
        loop.set_postfix({"Loss(training)": train_history['loss'][-1], "Loss(test)": test_history['loss'][-1]})
        
        if test_loss < min_test_loss:
            min_test_loss = test_loss

    # Save the trained models
    if not os.path.exists(HyperParams.net_dir):
        os.makedirs(HyperParams.net_dir)

    torch.save(dyn_model.state_dict(), HyperParams.net_dir + HyperParams.net_name + '_dyn.pt')
    torch.save(rec_model.state_dict(), HyperParams.net_dir + HyperParams.net_name + '_rec.pt')

    return train_history, test_history


def train_one_epoch(dyn_model, rec_model, device, params_train, times,\
                    data, position_dataset, train_snapshots, HyperParams):
    train_loss = 0
    num_integrations = round(float((times[1]))/HyperParams.dt)  # How many times we need to integrate to reach the next snapshot
    dyn_model.train()
    rec_model.train()
    #stn = torch.zeros(HyperParams.dim_latent, device=device)
    optimizer = torch.optim.LBFGS(list(dyn_model.parameters()) + list(rec_model.parameters()))
    for (alpha_indx, alpha) in enumerate(params_train):
        
        def closure():
            optimizer.zero_grad()
            stn_vec = []
            t_integration = []
            stn = torch.zeros(HyperParams.dim_latent, device=device)
            stn_vec.append(stn)
            t_integration.append(times[0])
            tot_pos_batches = len(position_dataset) // HyperParams.batch_pos_size
            pos_batch_sampler = np.random.randint(0, tot_pos_batches, HyperParams.num_pos_batches)  

            for (t_indx, t) in enumerate(times[1:]):
                for j in range(num_integrations):
                    # Get u(t)
                    u_t_needed = gaussian_process.eval_u_t(t_integration[-1], params_train[alpha_indx], HyperParams.T_f)
                    u_t_needed = torch.tensor(u_t_needed, device=device)
                    # Forward pass in DynNet
                    dyn_input = torch.cat((u_t_needed.unsqueeze(0), stn), dim=0)
                    stn_derivative = dyn_model(dyn_input)
                    # Forward Euler
                    stn = stn + HyperParams.dt * stn_derivative
                    stn_vec.append(stn)
                    # Keep track of the current time
                    t_integration.append(t_integration[-1] + HyperParams.dt)

                for j in pos_batch_sampler:
                    pos_indices = np.array(range(j*HyperParams.batch_pos_size, (j+1)*HyperParams.batch_pos_size))
                    x_pos, y_pos = position_dataset[pos_indices, 0], position_dataset[pos_indices, 1]
                    rec_input = torch.cat((stn, x_pos.to(device), y_pos.to(device)), dim=0)
                    velocity_pred = rec_model(rec_input)
                    velocity_target = []
                    for i in range(data.size(2)):
                        velocity_comp_target = data[(alpha_indx*(len(times)-1)+t_indx), pos_indices, i]
                        velocity_target.append(velocity_comp_target)
                    velocity_target = torch.cat(velocity_target, dim=0)
                    loss_rec = F.mse_loss(velocity_pred, velocity_target, reduction="mean")
                    train_loss += loss_rec.item()
                    loss_rec.backward(retain_graph=True)
            return train_loss

        optimizer.step(closure)

    return train_loss / (len(params_train)*len(times)*HyperParams.num_pos_batches)


def evaluate_model(dyn_model, rec_model, device, params_test, times, data, position_dataset, test_snapshots, HyperParams):
    test_loss = 0
    num_integrations = round(float((times[1]))/HyperParams.dt)  
    dyn_model.eval()
    rec_model.eval()
    stn = torch.zeros(HyperParams.dim_latent, device=device)

    with torch.no_grad():

        for (alpha_indx,alpha) in enumerate(params_test):
            stn_vec = []
            t_integration = []
            stn = torch.zeros(HyperParams.dim_latent, device=device)
            stn_vec.append(stn)
            t_integration.append(times[0])
            tot_pos_batches = len(position_dataset) // HyperParams.batch_pos_size
            pos_batch_sampler = np.random.randint(0, tot_pos_batches, HyperParams.num_pos_batches) 

            for (t_indx, t) in enumerate(times[1:]):

                for j in range(num_integrations):
                    u_t_needed = gaussian_process.eval_u_t(t_integration[-1], params_test[alpha_indx], HyperParams.T_f)
                    u_t_needed = torch.tensor(u_t_needed, device=device)
                    dyn_input = torch.cat((u_t_needed.unsqueeze(0), stn), dim=0)
                    stn_derivative = dyn_model(dyn_input)
                    stn = stn + HyperParams.dt * stn_derivative
                    stn_vec.append(stn)
                    t_integration.append(t_integration[-1] + HyperParams.dt)
                
                for j in pos_batch_sampler:
                    pos_indices = np.array(range(j*HyperParams.batch_pos_size, (j+1)*HyperParams.batch_pos_size))
                    x_pos, y_pos = position_dataset[pos_indices, 0], position_dataset[pos_indices, 1]
                    rec_input = torch.cat((stn, x_pos.to(device), y_pos.to(device)), dim=0)
                    velocity_pred = rec_model(rec_input)
                    velocity_target = []
                    for i in range(data.size(2)):
                        velocity_comp_target = data[(alpha_indx*(len(times)-1)+t_indx), pos_indices, i]
                        velocity_target.append(velocity_comp_target)
                    velocity_target = torch.cat(velocity_target, dim=0)
                    loss_rec = F.mse_loss(velocity_pred, velocity_target, reduction="mean")
                    test_loss += loss_rec.item()

    return test_loss / (len(params_test)*len(times)*HyperParams.num_pos_batches)