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
def train_dyn_rec_nets(dyn_model, rec_model, optimizer, scheduler, device,\
                        params_train, params_test, times, train_loader, test_loader, position_dataset, HyperParams):
    
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
                                      times, data_train, position_dataset, HyperParams)

        train_history["loss"].append(train_loss)          
        test_loss = evaluate_model(dyn_model, rec_model, device, params_test, \
                                   times, data_test, position_dataset,  HyperParams)
    
        test_history["loss"].append(test_loss)
        # Update the learning rate
        scheduler.step()
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


# Function to train for one epoch
def train_one_epoch(dyn_model, rec_model, optimizer, device, params_train, times,\
                    data, position_dataset, HyperParams):
    train_loss = 0
    num_integrations = round(float((times[1]))/HyperParams.dt)  # How many times we need to integrate to reach the next snapshot
    dyn_model.train()
    rec_model.train()
    #stn = torch.zeros(HyperParams.dim_latent, device=device)
    optimizer.zero_grad()
    for (alpha_indx, alpha) in enumerate(params_train):
        
        stn_vec = []
        t_integration = []
        stn = torch.zeros(HyperParams.dim_latent, device=device).detach()
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

            # Generate HyperParams.num_pos_batches random number between 0 and tot_pos_batches
            # These numbers will be the indices of the batches of positions that will be used for training
            loss_rec = 0 
            # Take a batch of positions
            for j in pos_batch_sampler:
                pos_indices = np.array(range(j*HyperParams.batch_pos_size, (j+1)*HyperParams.batch_pos_size))
                x_pos, y_pos = position_dataset[pos_indices, 0], position_dataset[pos_indices, 1]
                velocity_preds = []
                velocity_targets = []
                # Take each pair (x,y)
                counter = 0
               
                for x, y in zip(x_pos, y_pos):
                    # Unsqueeze x and y
                    x = x.unsqueeze(0).detach()
                    y = y.unsqueeze(0).detach()
                    rec_input = torch.cat((stn, x.to(device), y.to(device)), dim=0)
                    velocity_pred = rec_model(rec_input)
                    # Take the target velocities vx and vy
                    velocity_target = []
                    for i in range(data.size(2)):
                        velocity_comp_target = data[(alpha_indx*(len(times)-1)+t_indx), pos_indices[counter], i]
                        velocity_comp_target = velocity_comp_target.unsqueeze(0)
                        velocity_target.append(velocity_comp_target)
                    velocity_target = torch.cat(velocity_target, dim=0)

                    loss_rec += F.mse_loss(velocity_pred, velocity_target, reduction="mean")
                    train_loss += loss_rec.item()
                    
                    counter = counter+1
            loss_rec.backward(retain_graph=True)

        
        optimizer.step()
        optimizer.zero_grad()  
           
            

                    # Take the velocities of the right snapshot of the series that defines the current simulation
                #     velocity_target = []
                    
                #     velocity_targets.append(torch.cat(velocity_target, dim=0))
                
                # velocity_preds = torch.stack(velocity_preds)
                # velocity_targets = torch.stack(velocity_targets)
    
                  
        
   
    return train_loss / (len(params_train)*len(times)*HyperParams.num_pos_batches*HyperParams.batch_pos_size )




def evaluate_model(dyn_model, rec_model, device, params_eval, times, data, position_dataset, HyperParams):
    eval_loss = 0
    num_integrations = round(float((times[1]))/HyperParams.dt)
    dyn_model.eval()
    rec_model.eval()
    with torch.no_grad():
        for (alpha_indx, alpha) in enumerate(params_eval):
            stn_vec = []
            t_integration = []
            stn = torch.zeros(HyperParams.dim_latent, device=device)
            stn_vec.append(stn)
            t_integration.append(times[0])
            tot_pos_batches = len(position_dataset) // HyperParams.batch_pos_size
            pos_batch_sampler = np.random.randint(0, tot_pos_batches, HyperParams.num_pos_batches)  

            for (t_indx, t) in enumerate(times[1:]):
                for j in range(num_integrations):
                    u_t_needed = gaussian_process.eval_u_t(t_integration[-1], params_eval[alpha_indx], HyperParams.T_f)
                    u_t_needed = torch.tensor(u_t_needed, device=device)
                    dyn_input = torch.cat((u_t_needed.unsqueeze(0), stn), dim=0)
                    stn_derivative = dyn_model(dyn_input)
                    stn = stn + HyperParams.dt * stn_derivative
                    stn_vec.append(stn)
                    t_integration.append(t_integration[-1] + HyperParams.dt)

                loss_rec = 0
                for j in pos_batch_sampler:
                    pos_indices = np.array(range(j*HyperParams.batch_pos_size, (j+1)*HyperParams.batch_pos_size))
                    x_pos, y_pos = position_dataset[pos_indices, 0], position_dataset[pos_indices, 1]
                    counter = 0
                    for x, y in zip(x_pos, y_pos):
                        x = x.unsqueeze(0)
                        y = y.unsqueeze(0)
                        rec_input = torch.cat((stn, x.to(device), y.to(device)), dim=0)
                        velocity_pred = rec_model(rec_input)
                        velocity_target = []
                        for i in range(data.size(2)):
                            velocity_comp_target = data[(alpha_indx*(len(times)-1)+t_indx), pos_indices[counter], i]
                            velocity_comp_target = velocity_comp_target.unsqueeze(0)
                            velocity_target.append(velocity_comp_target)
                        velocity_target = torch.cat(velocity_target, dim=0)

                        loss_rec += F.mse_loss(velocity_pred, velocity_target, reduction="mean")
                        eval_loss += loss_rec.item()
                        counter += 1

    return eval_loss / (len(params_eval)*len(times)*HyperParams.num_pos_batches*HyperParams.batch_pos_size )