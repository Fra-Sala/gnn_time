import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import logging

torch.set_default_dtype(torch.float32)

def train_dyn_rec_nets(dyn_model, rec_model, dyn_optimizer, rec_optimizer, dyn_scheduler, rec_scheduler, device, u_t, times, train_loader, test_loader, position_loader, train_snapshots, test_snapshots, HyperParams):
    logging.basicConfig(level=logging.INFO)
    train_history = {"loss": [], "l1": [], "l2": []}
    test_history = {"loss": [], "l1": [], "l2": []}
    min_test_loss = np.Inf

    n_sim_train = len(train_snapshots)
    u_t = u_t.to(torch.float32)
    u_t_train = u_t[train_snapshots]
    u_t_test = u_t[test_snapshots]

    dyn_model.to(torch.float32)
    rec_model.to(torch.float32)
    dyn_model.to(device)
    rec_model.to(device)
    dyn_model.train()
    rec_model.train()
    
    # Set up the progress bar
    loop = tqdm(range(HyperParams.max_epochs))

    for epoch in loop:
        train_loss = train_one_epoch(dyn_model, rec_model, dyn_optimizer, rec_optimizer, dyn_scheduler, rec_scheduler, device, u_t_train, times, train_loader, position_loader, train_snapshots, HyperParams)
        train_history["loss"].append(train_loss)

        # Evaluate the model on the test set and store the results in test_history
        test_loss = evaluate_model(dyn_model, rec_model, device, u_t_test, times, test_loader, position_loader, test_snapshots, HyperParams)
        test_history["loss"].append(test_loss)

        # Log the results
        logging.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        if test_loss < min_test_loss:
            # Save the model
            min_test_loss = test_loss

    # save the trained models 
    # ...

def train_one_epoch(dyn_model, rec_model, dyn_optimizer, rec_optimizer, dyn_scheduler, rec_scheduler, device, u_t_train, times, train_loader, position_loader, train_snapshots, HyperParams):
    train_loss = 0
    for sim_indx in train_snapshots:
        dyn_optimizer.zero_grad()
        rec_optimizer.zero_grad()
        data_iterator = iter(train_loader)
        data = next(data_iterator).to(device)
        data.to(torch.float32)

        stn = torch.zeros(HyperParams.dim_latent, device=device, dtype=torch.float32)

        for t in times:
            print("New time: ", t)
            dyn_input = torch.cat((u_t_train[sim_indx].unsqueeze(0), stn), dim=0)
            stn_derivative = dyn_model(dyn_input)

            stn_plus_one = stn + HyperParams.dt * stn_derivative

            for j, pos in enumerate(position_loader):
                x_pos, y_pos = pos
                if(len(x_pos) == HyperParams.batch_size_pos):
                    rec_input = torch.cat((stn_plus_one, x_pos, y_pos), dim=0)
                    velocity_pred = rec_model(rec_input)
                    velocity_target = data[:, len(x_pos) * j : len(x_pos) * (j + 1), 0]
                    loss_rec = F.mse_loss(velocity_pred, velocity_target, reduction="mean")
                    loss_rec.backward(retain_graph=True)
            print("I have gone through all the positions")
            stn = stn_plus_one
            train_loss += loss_rec.item()
            
        # Update weights after each simulation
        rec_optimizer.step()
        dyn_optimizer.step()
        rec_scheduler.step()
        dyn_scheduler.step()

    return train_loss / len(train_snapshots)

def evaluate_model(dyn_model, rec_model, device, u_t_test, times, test_loader, position_loader, test_snapshots, HyperParams):
    test_loss = 0
    with torch.no_grad():
        for sim_indx in test_snapshots:
            data_iterator = iter(test_loader)
            data = next(data_iterator).to(device)
            stn = torch.zeros(HyperParams.dim_latent, device=device)
            stn.to(torch.float32)
            
            for t in times:
                dyn_input = torch.cat((u_t_test[sim_indx].unsqueeze(0), stn), dim=0)
                stn_derivative = dyn_model(dyn_input)
                stn_plus_one = (stn + HyperParams.dt * stn_derivative).to(torch.float32)

                
                for j, pos in enumerate(position_loader):
                    x_pos, y_pos = pos
                    rec_input = torch.cat((stn_plus_one, x_pos, y_pos), dim=0)
                    velocity_pred = rec_model(rec_input)
                    velocity_target = data[:, len(x_pos) * j : len(x_pos) * (j + 1), 0]
                    test_loss += F.mse_loss(velocity_pred, velocity_target, reduction="mean").item()
                stn = stn_plus_one

    return test_loss / len(test_snapshots)

############ OLD IMPLEMENTATION
# def train_dyn_rec_nets(dyn_model, rec_model, dyn_optimizer, rec_optimizer, dyn_scheduler,\
#                         rec_scheduler, device, u_t, times, train_loader, test_loader, position_loader, train_snapshots, test_snapshots, HyperParams):
#     train_history = dict(train=[], l1=[], l2=[])
#     test_history = dict(test=[], l1=[], l2=[])
#     min_test_loss = np.Inf
#     n_sim_train = len(train_snapshots)
#     u_t = u_t.to(torch.float32)
#     u_t_train = u_t[train_snapshots]
#     u_t_test = u_t[test_snapshots]
#     dyn_model.to(torch.float32)
#     rec_model.to(torch.float32)
#     dyn_model.train()
#     rec_model.train()
#     torch.autograd.set_detect_anomaly(True)
#     loop = tqdm(range(HyperParams.max_epochs))
    
#     for epoch in loop:
#         train_rmse = total_examples = sum_loss = 0
#         train_rmse_1 = train_rmse_2 = 0
#         sum_loss_1 = sum_loss_2 = 0
#         i = 0        

#         for sim_indx in train_snapshots:  # get the number of snapshot
#             dyn_optimizer.zero_grad()
#             rec_optimizer.zero_grad()

#             data_iterator = iter(train_loader)
#             # Get the data
#             data = next(data_iterator).to(device)
#             #data = data.to(torch.float32)

#             # The initial latent state is an array of zeros
#             stn = torch.zeros(HyperParams.dim_latent, device=device)
           
#             stn = stn.to(torch.float32)

#             print(sim_indx)
                    
#             for t in times:
#                 # Forward pass through DynNet
#                 dyn_input = torch.cat((u_t[sim_indx].unsqueeze(0), stn), dim=0)
#                 stn_derivative = dyn_model(dyn_input)

#                 print("time: ", t)
#                 # Compute s(tn+1) using forward Euler method
#                 stn_plus_one = stn + HyperParams.dt * stn_derivative
#                 stn_plus_one = stn_plus_one.to(torch.float32)
#                 j = 0
#                 for pos in position_loader:
#                     x_pos, y_pos = pos
#                     x_pos = x_pos.to(torch.float32)
#                     y_pos = y_pos.to(torch.float32)
#                     # Forward pass through RecNet
#                     rec_input = torch.cat((stn_plus_one, x_pos, y_pos), dim=0)
#                     rec_input = rec_input.to(torch.float32)
#                     velocity_pred = rec_model(rec_input)

#                     # Calculate the loss (i-th training simulation, take 10 positions, scalar)
#                     velocity_target = data[i, len(x_pos)*j:len(x_pos)*(j+1), 0]
#                     #velocity_pred = velocity_pred.to(torch.float32)
#                     #velocity_target = velocity_target.to(torch.float32)
#                     loss_rec = F.mse_loss(velocity_pred, velocity_target, reduction='mean')
                   
#                     # Backpropagation and parameter updates for both models
#                     loss_rec.backward(retain_graph =True)
                    
#                     j=j+1

#                 # Update s(t) for the next time step
#                 stn = stn_plus_one

#                 sum_loss += loss_rec.item()
#                 total_examples += 1

#             rec_optimizer.step()
#             dyn_optimizer.step()
#             rec_scheduler.step()
#             dyn_scheduler.step()

#             i = i+1
########## THE CODE HAS BEEN UPDATED ONLY UP UNTIL HERE

    #     train_rmse = sum_loss / total_examples
    #     train_rmse_1 = sum_loss_1 / total_examples
    #     train_rmse_2 = sum_loss_2 / total_examples
    #     train_history['train'].append(train_rmse)
    #     train_history['l1'].append(train_rmse_1)
    #     train_history['l2'].append(train_rmse_2)

    #     if HyperParams.cross_validation:
    #         with torch.no_grad():
    #             dyn_model.eval()
    #             rec_model.eval()
    #             test_rmse = total_examples = sum_loss = 0
    #             test_rmse_1 = test_rmse_2 = 0
    #             sum_loss_1 = sum_loss_2 = 0

    #             for data in test_loader:
    #                 data = data.to(device)
    #                 stn = torch.zeros(HyperParams.dim_latent, device=device)
    #                 for t in range(data.size(1)):
    #                     u_t = data[:, t, :2]
    #                     position_x = data[:, t, 2]
    #                     position_y = data[:, t, 3]
                        
    #                     dyn_input = torch.cat((u_t, stn), dim=1)
    #                     stn_derivative = dyn_model(dyn_input)
    #                     stn_plus_one = stn + HyperParams.dt * stn_derivative
                        
    #                     rec_input = torch.cat((stn_plus_one, position_x, position_y), dim=1)
    #                     velocity_pred = rec_model(rec_input)
                        
    #                     target_velocity = data[:, t, 4]
    #                     loss_rec = F.mse_loss(velocity_pred, target_velocity, reduction='mean')
                        
    #                     loss_dyn = F.mse_loss(stn_plus_one, stn, reduction='mean')
    #                     loss_test = loss_dyn + HyperParams.lambda_rec * loss_rec
                        
    #                     sum_loss += loss_test.item()
    #                     sum_loss_1 += loss_dyn.item()
    #                     sum_loss_2 += loss_rec.item()
    #                     total_examples += 1

    #             test_rmse = sum_loss / total_examples
    #             test_rmse_1 = sum_loss_1 / total_examples
    #             test_rmse_2 = sum_loss_2 / total_examples
    #             test_history['test'].append(test_rmse)
    #             test_history['l1'].append(test_rmse_1)
    #             test_history['l2'].append(test_rmse_2)
                
    #         loop.set_postfix({"Loss(training)": train_history['train'][-1], "Loss(validation)": test_history['test'][-1]})

    #     else:
    #         test_rmse = train_rmse
    #         loop.set_postfix({"Loss(training)": train_history['train'][-1]})

    #     if test_rmse < min_test_loss:
    #         min_test_loss = test_rmse
    #         best_epoch = epoch
    #         # Save the models if needed
    #         torch.save(dyn_model.state_dict(), HyperParams.dyn_model_path)
    #         torch.save(rec_model.state_dict(), HyperParams.rec_model_path)
        
    #     if HyperParams.tolerance >= train_rmse:
    #         print('Early stopping!')
    #         break

    #     np.save(HyperParams.train_history_path, train_history)
    #     np.save(HyperParams.test_history_path, test_history)
    
    # print("\nLoading best models for epoch: ", best_epoch)
    # dyn_model.load_state_dict(torch.load(HyperParams.dyn_model_path, map_location=torch.device('cpu')))
    # rec_model.load_state_dict(torch.load(HyperParams.rec_model_path, map_location=torch.device('cpu')))
########## END OF OLD IMPLEMENTATION

# def train_dyn_rec_nets(dyn_model, rec_model, dyn_optimizer, rec_optimizer, dyn_scheduler, rec_scheduler, device, ut,times, train_loader, test_loader, position_loader, HyperParams):
#     train_history = dict(train=[], l1=[], l2=[])
#     test_history = dict(test=[], l1=[], l2=[])
#     min_test_loss = np.Inf

#     dyn_model.train()
#     rec_model.train()
#     loop = tqdm(range(HyperParams.max_epochs))
    
#     for epoch in loop:
#         train_rmse = total_examples = sum_loss = 0
#         train_rmse_1 = train_rmse_2 = 0
#         sum_loss_1 = sum_loss_2 = 0
#         i = 0
#         sim_indx = 0

#         for data in train_loader:
#             dyn_optimizer.zero_grad()
#             rec_optimizer.zero_grad()
#             data = data.to(device) 
                    
#             # The initial latent state is an array of zeros
#             stn = torch.zeros(HyperParams.dim_latent, device=device)
            
#             for t in range(times):

#                 # Forward pass through DynNet
#                 dyn_input = torch.cat((ut[i*sim_indx], stn), dim=1)
#                 stn_derivative = dyn_model(dyn_input)

#                 # Compute s(tn+1) using forward Euler method
#                 stn_plus_one = stn + HyperParams.dt * stn_derivative

#                 for pos in position_loader:
#                     # Use 10 positions (x,y) at a time
#                     x_pos, y_pos = pos
#                     # Forward pass through RecNet
#                     rec_input = torch.cat((stn_plus_one, x_pos, y_pos), dim=1)
#                     velocity_pred = rec_model(rec_input)

#                     # Calculate the loss
#                     velocity_target = data[len(x_pos)*i,sim_indx:, :]
#                     loss_rec = F.mse_loss(velocity_pred, velocity_target, reduction='mean')

#                     # Backpropagation and parameter updates for both models
#                     loss_rec.backward()
#                     rec_optimizer.step()
#                     dyn_optimizer.step()
#                     rec_scheduler.step()
#                     dyn_scheduler.step()

#                 # Update s(t) for the next time step
#                 stn = stn_plus_one

#                 sum_loss += loss_rec.item()
#                 total_examples += 1

#         train_rmse = sum_loss / total_examples
#         train_rmse_1 = sum_loss_1 / total_examples
#         train_rmse_2 = sum_loss_2 / total_examples
#         train_history['train'].append(train_rmse)
#         train_history['l1'].append(train_rmse_1)
#         train_history['l2'].append(train_rmse_2)

#         if HyperParams.cross_validation:
#             with torch.no_grad():
#                 dyn_model.eval()
#                 rec_model.eval()
#                 test_rmse = total_examples = sum_loss = 0
#                 test_rmse_1 = test_rmse_2 = 0
#                 sum_loss_1 = sum_loss_2 = 0

#                 for data in test_loader:
#                     data = data.to(device)
#                     stn = torch.zeros(HyperParams.dim_latent, device=device)
#                     for t in range(data.size(1)):
#                         u_t = data[:, t, :2]
#                         position_x = data[:, t, 2]
#                         position_y = data[:, t, 3]
                        
#                         dyn_input = torch.cat((u_t, stn), dim=1)
#                         stn_derivative = dyn_model(dyn_input)
#                         stn_plus_one = stn + HyperParams.dt * stn_derivative
                        
#                         rec_input = torch.cat((stn_plus_one, position_x, position_y), dim=1)
#                         velocity_pred = rec_model(rec_input)
                        
#                         target_velocity = data[:, t, 4]
#                         loss_rec = F.mse_loss(velocity_pred, target_velocity, reduction='mean')
                        
#                         loss_dyn = F.mse_loss(stn_plus_one, stn, reduction='mean')
#                         loss_test = loss_dyn + HyperParams.lambda_rec * loss_rec
                        
#                         sum_loss += loss_test.item()
#                         sum_loss_1 += loss_dyn.item()
#                         sum_loss_2 += loss_rec.item()
#                         total_examples += 1

#                 test_rmse = sum_loss / total_examples
#                 test_rmse_1 = sum_loss_1 / total_examples
#                 test_rmse_2 = sum_loss_2 / total_examples
#                 test_history['test'].append(test_rmse)
#                 test_history['l1'].append(test_rmse_1)
#                 test_history['l2'].append(test_rmse_2)
                
#             loop.set_postfix({"Loss(training)": train_history['train'][-1], "Loss(validation)": test_history['test'][-1]})

#         else:
#             test_rmse = train_rmse
#             loop.set_postfix({"Loss(training)": train_history['train'][-1]})

#         if test_rmse < min_test_loss:
#             min_test_loss = test_rmse
#             best_epoch = epoch
#             # Save the models if needed
#             torch.save(dyn_model.state_dict(), HyperParams.dyn_model_path)
#             torch.save(rec_model.state_dict(), HyperParams.rec_model_path)
        
#         if HyperParams.tolerance >= train_rmse:
#             print('Early stopping!')
#             break

#         np.save(HyperParams.train_history_path, train_history)
#         np.save(HyperParams.test_history_path, test_history)
    
#     print("\nLoading best models for epoch: ", best_epoch)
#     dyn_model.load_state_dict(torch.load(HyperParams.dyn_model_path, map_location=torch.device('cpu')))
#     rec_model.load_state_dict(torch.load(HyperParams.rec_model_path, map_location=torch.device('cpu')))
