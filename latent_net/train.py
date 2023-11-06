import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Set default data type to float32
torch.set_default_dtype(torch.float32)

# Function to train dynamic and rec networks
def train_dyn_rec_nets(dyn_model, rec_model, dyn_optimizer, rec_optimizer, dyn_scheduler, rec_scheduler, device, u_t, times, train_loader, test_loader, position_loader, train_snapshots, test_snapshots, HyperParams):
    
    train_history = {"loss": [], "l1": [], "l2": []}
    test_history = {"loss": [], "l1": [], "l2": []}
    min_test_loss = np.Inf
    u_t = u_t.to(torch.float32)

    dyn_model.to(torch.float32)
    rec_model.to(torch.float32)
    dyn_model.to(device)
    rec_model.to(device)
    dyn_model.train()
    rec_model.train()
    
    # Progress bar
    loop = tqdm(range(HyperParams.max_epochs))

    for epoch in loop:
        train_loss = train_one_epoch(dyn_model, rec_model, dyn_optimizer, rec_optimizer, dyn_scheduler, rec_scheduler, device, u_t, times, train_loader, position_loader, train_snapshots, HyperParams)
        train_history["loss"].append(train_loss)

        # Evaluate the model on the test set and store the results in test_history
        test_loss = evaluate_model(dyn_model, rec_model, device, u_t, times, test_loader, position_loader, test_snapshots, HyperParams)
        test_history["loss"].append(test_loss)

        # Print the results
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        if test_loss < min_test_loss:

            min_test_loss = test_loss

    # Save the trained models
    # ...

# Function to train for one epoch
def train_one_epoch(dyn_model, rec_model, dyn_optimizer, rec_optimizer, dyn_scheduler, rec_scheduler, device, u_t, times, train_loader, position_loader, train_snapshots, HyperParams):
    train_loss = 0
    for (n_sim, sim_indx) in enumerate(train_snapshots):
        dyn_optimizer.zero_grad()
        rec_optimizer.zero_grad()
        data_iterator = iter(train_loader)
        data = next(data_iterator).to(device)
        data.to(torch.float32)

        st_vec = []
        t_integration = []

        stn = torch.zeros(HyperParams.dim_latent, device=device, dtype=torch.float32)
        # Keep track of s(t) and times of integration
        st_vec.append(stn)
        t_integration.append(times[0])
        
        for t in times:
            t.float()
            
            # To perform linear interpolation, retrieve the sequence u(t)
            sequence_length = len(times)
            start_index = (sim_indx // sequence_length) * sequence_length
            end_index = start_index + sequence_length
            current_u_t = u_t[start_index:end_index]

            # Keep making the forward step in the DynNet until we know s(t) up to the next value of t in the array times
            while t_integration[-1] <= (t + times[-1] - times[-2]):
                # Compute u(t*), where t* is the time for which we know s(t*).
                u_t_needed = torch.tensor(np.interp(np.array(t_integration[-1]), times, current_u_t), dtype=torch.float32)

                dyn_input = torch.cat((u_t_needed.unsqueeze(0), stn), dim=0)
                stn_derivative = dyn_model(dyn_input)

                stn_plus_one = stn + HyperParams.dt * stn_derivative
                st_vec.append(stn_plus_one)
                t_integration.append(t_integration[-1] + HyperParams.dt)

            for j, pos in enumerate(position_loader):
                x_pos, y_pos = pos

                if len(x_pos) == HyperParams.batch_size_pos:
                    # For the RecNet, use the last stn computed at time t \in times
                    index = int((t_integration[-1] - t) / (HyperParams.dt))
                    stn_needed = st_vec[-index-1]

                    rec_input = torch.cat((stn_needed, x_pos, y_pos), dim=0)
                    velocity_pred = rec_model(rec_input)
                    velocity_target = data[n_sim, HyperParams.batch_size_pos * j: HyperParams.batch_size_pos * (j + 1), 0]
                    loss_rec = F.mse_loss(velocity_pred, velocity_target, reduction="mean")

            stn = stn_plus_one
            train_loss += loss_rec.item()

        # Backpropagate
        loss_rec.backward()
        # Update weights after each simulation
        rec_optimizer.step()
        dyn_optimizer.step()
        rec_scheduler.step()
        dyn_scheduler.step()

    return train_loss / len(train_snapshots)

# Function to evaluate the model on the test set
def evaluate_model(dyn_model, rec_model, device, u_t, times, test_loader, position_loader, test_snapshots, HyperParams):
    test_loss = 0
    with torch.no_grad():
        for (n_sim, sim_indx) in enumerate(test_snapshots):
            data_iterator = iter(test_loader)
            data = next(data_iterator).to(device)
            data = data.to(torch.float32)
            stn = torch.zeros(HyperParams.dim_latent, device=device, dtype=torch.float32)

            st_vec = []
            t_integration = []

            stn = torch.zeros(HyperParams.dim_latent, device=device, dtype=torch.float32)
            # Keep track of s(t) and times of integration
            st_vec.append(stn)
            t_integration.append(times[0])

            for t in times:
                t.float()
                sequence_length = len(times)
                start_index = (sim_indx // sequence_length) * sequence_length
                end_index = start_index + sequence_length
                current_u_t = u_t[start_index:end_index]

                dyn_input = torch.cat((u_t[sim_indx].unsqueeze(0), stn), dim=0)
                dyn_input = dyn_input.to(torch.float32)
                stn_derivative = dyn_model(dyn_input)
                stn_plus_one = (stn + HyperParams.dt * stn_derivative).to(torch.float32)

                # Add while loop to match training behavior
                while t_integration[-1] <= (t + times[-1] - times[-2]):
                    u_t_needed = torch.tensor(np.interp(np.array(t_integration[-1]), times, current_u_t), dtype=torch.float32)
                    dyn_input = torch.cat((u_t_needed.unsqueeze(0), stn), dim=0)
                    stn_derivative = dyn_model(dyn_input)
                    stn_plus_one = stn + HyperParams.dt * stn_derivative
                    t_integration.append(t_integration[-1] + HyperParams.dt)

                for j, pos in enumerate(position_loader):
                    x_pos, y_pos = pos

                    if len(x_pos) == HyperParams.batch_size_pos:
                        rec_input = torch.cat((stn_plus_one, x_pos, y_pos), dim=0)
                        velocity_pred = rec_model(rec_input)
                        velocity_target = data[n_sim, HyperParams.batch_size_pos * j: HyperParams.batch_size_pos * (j + 1), 0]
                        test_loss += F.mse_loss(velocity_pred, velocity_target, reduction="mean").item()
                stn = stn_plus_one

    return test_loss / len(test_snapshots)
