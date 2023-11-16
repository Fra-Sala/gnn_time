import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from latent_net import shuffle_dataset
import time
import os
from lid_driven_cavity_fenics import gaussian_process

# Set default data type to float32
#torch.set_default_dtype(torch.float32)

# Function to train dynamic and rec networks
def train_dyn_rec_nets(dyn_model, rec_model, optimizer, scheduler, device,\
                        alpha1_tensor, times, train_loader, test_loader, position_dataset, train_snapshots, test_snapshots, HyperParams):
    
    train_history = {"loss": [], "l1": [], "l2": []}
    test_history = {"loss": [], "l1": [], "l2": []}
    min_test_loss = np.Inf
    # u_t = u_t.to(torch.float32)



    # dyn_model.to(torch.float32)
    # rec_model.to(torch.float32)
    dyn_model.to(device)
    rec_model.to(device)
    dyn_model.train()
    rec_model.train()
    
    # Progress bar
    loop = tqdm(range(HyperParams.max_epochs))

    for epoch in loop:

        # Shuffle the batches of positions
        pos_batch_sampler, position_loader = shuffle_dataset.shuffle_position_dataset(position_dataset, HyperParams, seed=epoch)

        dyn_model.train()
        rec_model.train()
        train_loss = train_one_epoch(dyn_model, rec_model, optimizer, scheduler, device, alpha1_tensor,\
                                      times, train_loader, position_loader, train_snapshots, HyperParams, pos_batch_sampler)
        train_history["loss"].append(train_loss)

        # Evaluate the model on the test set and store the results in test_history
        test_loss = evaluate_model(dyn_model, rec_model, device, alpha1_tensor, times, test_loader, position_loader, test_snapshots, HyperParams,\
                                    pos_batch_sampler)
        test_history["loss"].append(test_loss)

        # Print the results
        loop.set_postfix({"Loss(training)": train_history['loss'][-1], "Loss(test)": test_history['loss'][-1]})
        #print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        if test_loss < min_test_loss:

            min_test_loss = test_loss

    # Save the trained models
    if not os.path.exists(HyperParams.net_dir):
        os.makedirs(HyperParams.net_dir)
    torch.save(dyn_model.state_dict(), HyperParams.net_dir + HyperParams.net_name + '_dyn.pt')
    torch.save(rec_model.state_dict(), HyperParams.net_dir + HyperParams.net_name + '_rec.pt')

    return train_history, test_history


# Function to train for one epoch
def train_one_epoch(dyn_model, rec_model, optimizer, scheduler, device, alpha1_tensor, times,\
                     train_loader, position_loader, train_snapshots, HyperParams, pos_batch_sampler):
    train_loss = 0
    dyn_model.to(device)
    rec_model.to(device)
    
    data_iterator = iter(train_loader)
    data = next(data_iterator).to(device)
    for (n_snap, snap) in enumerate(train_snapshots):
       
        optimizer.zero_grad()
        stn_vec = []
        t_integration = []
        stn = torch.zeros(HyperParams.dim_latent, device=device)
        stn_vec.append(stn)
        t_integration.append(times[0])
        
        for (k,t) in enumerate(times):

            # Old approach: retrieve the sequence u(t) to which the current snapshot belongs
            # sequence_length = len(times)
            # start_index = (snap // sequence_length) * sequence_length
            # end_index = start_index + sequence_length
            # current_u_t = u_t[start_index:end_index].to(device)

            # New approach: retrieve the Fourier mode in alpha1_vector to which the current snapshot belongs
            # There are only 20 fourier modes in alpha1_vector, so every 9 snapshots belong to the same mode
            # Retrieve the correct Fourier mode
            index_alpha = snap // 9            
            number_of_integrations = round(float((times[0]))/HyperParams.dt)

            if t != times[-1]:
                for j in range(number_of_integrations):
                    #u_t_needed = torch.tensor(np.interp(np.array(t_integration[-1]), times, current_u_t), device=device)
                    #print("The sequence u(t) is ", u_t_needed)
                    u_t_needed = gaussian_process.eval_u_t(t_integration[-1], alpha1_tensor[index_alpha], HyperParams.T_f)
                    u_t_needed = torch.tensor(u_t_needed, device=device)
                    # print("The snapshot is ", snap)
                    # print("the value of u_t is ", u_t_needed)
                    dyn_input = torch.cat((u_t_needed.unsqueeze(0), stn), dim=0)
                    stn_derivative = dyn_model(dyn_input)
                    stn = stn + HyperParams.dt * stn_derivative
                    stn_vec.append(stn)
                    t_integration.append(t_integration[-1] + HyperParams.dt)

            for j, pos in enumerate(position_loader):
                if j >= HyperParams.num_pos_batches:
                    break
                x_pos, y_pos = pos
                pos_indices = pos_batch_sampler[j]

                if len(x_pos) == HyperParams.batch_size_pos:
                    rec_input = torch.cat((stn, x_pos.to(device), y_pos.to(device)), dim=0)
                    velocity_pred = rec_model(rec_input)
                    velocity_target = data[n_snap, np.array(pos_indices), 0]
                    loss_rec = F.mse_loss(velocity_pred, velocity_target, reduction="mean")
                    loss_rec.backward(retain_graph=True)
                    train_loss += loss_rec.item()

            # if (k + 1) % 2 == 0 and k != len(times) - 1:
            #     for param in dyn_model.parameters():
            #         if param.grad is not None:
            #             param.grad = None 
            
        optimizer.step()

    return train_loss / len(train_snapshots)


def evaluate_model(dyn_model, rec_model, device, alpha1_tensor, times, test_loader, position_loader, test_snapshots, HyperParams, pos_batch_sampler):
    test_loss = 0
    dyn_model.to(device)
    rec_model.to(device)
    dyn_model.eval()
    rec_model.eval()
    data_iterator = iter(test_loader)
    data = next(data_iterator).to(device)
    with torch.no_grad():
        for (n_snap, snap) in enumerate(test_snapshots):
            

            stn_vec = []
            t_integration = []

            stn = torch.zeros(HyperParams.dim_latent, device=device)
            stn_vec.append(stn)
            t_integration.append(times[0])
            
            for (k,t) in enumerate(times):
                        
                index_alpha = snap // 9
          
                number_of_integrations = round(float((times[0]))/HyperParams.dt)

                if t != times[-1]:
                    for j in range(number_of_integrations):
                        #u_t_needed = torch.tensor(np.interp(np.array(t_integration[-1]), times, current_u_t), device=device)
                        #print("The sequence u(t) is ", u_t_needed)
                        u_t_needed = gaussian_process.eval_u_t(t_integration[-1], alpha1_tensor[index_alpha], HyperParams.T_f)
                        u_t_needed = torch.tensor(u_t_needed, device=device)

                        dyn_input = torch.cat((u_t_needed.unsqueeze(0), stn), dim=0)
                        stn_derivative = dyn_model(dyn_input)
                        stn = stn + HyperParams.dt * stn_derivative
                        stn_vec.append(stn)
                        t_integration.append(t_integration[-1] + HyperParams.dt)

                for j, pos in enumerate(position_loader):
                    if j >= HyperParams.num_pos_batches:
                        break
                    x_pos, y_pos = pos
                    pos_indices = pos_batch_sampler[j]

                    if len(x_pos) == HyperParams.batch_size_pos:
                        rec_input = torch.cat((stn, x_pos.to(device), y_pos.to(device)), dim=0)
                        velocity_pred = rec_model(rec_input)
                        velocity_target = data[n_snap, np.array(pos_indices), 0]
                        loss_rec = F.mse_loss(velocity_pred, velocity_target, reduction="mean")
                        
                        test_loss += loss_rec.item()



    return test_loss / len(test_snapshots)