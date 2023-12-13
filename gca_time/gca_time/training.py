import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm 
from gca_time import gaussian_process

def train(model_decoder, model_dyn, optimizer, device, scheduler, \
          train_loader, test_loader, HyperParams, params_train, params_test, times):
    """Trains latent-decoder model
    """
    train_history = dict(train=[])
    test_history = dict(test=[])
    min_test_loss = np.Inf
    
    model_decoder.train()
    model_dyn.train()
    loop = tqdm(range(HyperParams.max_epochs))
    for epoch in loop:
        train_rmse = total_examples = 0
        loss_train_mse = 0

        for (alpha_indx, alpha) in enumerate(params_train):
            stn_vec = []
            t_vec = []
            stn = torch.zeros(HyperParams.bottleneck_dim, device=device)
            stn_vec.append(stn)
            t_vec.append(times[0])

            for (t_indx, t) in enumerate(times[1:]):
                num_integrations = round(float((times[t_indx+1])-times[t_indx])/HyperParams.dt)
                u_t = []
                for j in range(num_integrations):
                    u_t = gaussian_process.eval_u_t(t_vec[-1], \
                                 params_train[alpha_indx], HyperParams.T_f)
                    u_t = torch.tensor(u_t, device=device)
                    dyn_input = torch.cat((u_t.unsqueeze(0), stn), dim=0)
                    stn_derivative = model_dyn(dyn_input)
                    stn = stn + HyperParams.dt * stn_derivative
                    t_vec.append(t_vec[-1] + HyperParams.dt)
                # Go through the decoder
                data = get_data(train_loader, (alpha_indx*(len(times)-1)+t_indx))
                data = data.to(device)
                decoder_input = torch.cat((u_t.unsqueeze(0), stn), dim=0)
                out = model_decoder(decoder_input, data)
                loss_train_mse += physics_loss(data.x, out, HyperParams)
                total_examples += 1
                
        loss_train_mse = loss_train_mse/total_examples
        optimizer.zero_grad()
        loss_train_mse.backward()      
        optimizer.step()
        scheduler.step()

        train_rmse = loss_train_mse
        train_history['train'].append(train_rmse.item())


        if HyperParams.cross_validation:
            with torch.no_grad():
                test_rmse = total_examples = sum_loss = 0
                loss_test_mse = 0
                for (alpha_indx, alpha) in enumerate(params_test):
                    stn_vec = []
                    t_vec = []
                    stn = torch.zeros(HyperParams.bottleneck_dim, device=device)
                    stn_vec.append(stn)
                    t_vec.append(times[0])

                    for (t_indx, t) in enumerate(times[1:]):
                        num_integrations = round(float((times[t_indx+1])-times[t_indx])/HyperParams.dt)
                        u_t = []
                        for j in range(num_integrations):
                            u_t = gaussian_process.eval_u_t(t_vec[-1], \
                                        params_test[alpha_indx], HyperParams.T_f)
                            u_t = torch.tensor(u_t, device=device)
                            dyn_input = torch.cat((u_t.unsqueeze(0), stn), dim=0)
                            stn_derivative = model_dyn(dyn_input)
                            stn = stn + HyperParams.dt * stn_derivative
                            t_vec.append(t_vec[-1] + HyperParams.dt)
                        # Go through the decoder
                        data = get_data(test_loader, (alpha_indx*(len(times)-1)+t_indx))
                        data = data.to(device)
                        decoder_input = torch.cat((u_t.unsqueeze(0), stn), dim=0)
                        out = model_decoder(decoder_input, data)
                        loss_test_mse += physics_loss(data.x, out, HyperParams)
                        total_examples += 1
                        
                loss_test_mse = loss_test_mse/total_examples
                test_rmse = loss_test_mse
                test_history['test'].append(loss_test_mse.item())


            # print("Epoch[{}/{}, train_mse loss:{}, test_mse loss:{}".format(epoch + 1, HyperParams.max_epochs, train_history['train'][-1], test_history['test'][-1]))
            loop.set_postfix({"Loss(training)": train_history['train'][-1], "Loss(validation)": test_history['test'][-1]})
        else:
            test_rmse = train_rmse
            # print("Epoch[{}/{}, train_mse loss:{}".format(epoch + 1, HyperParams.max_epochs, train_history['train'][-1]))
            loop.set_postfix({"Loss(training)": train_history['train'][-1]})

        # if test_rmse < min_test_loss:
        #     min_test_loss = test_rmse
        #     best_epoch = epoch
        #     torch.save(model_decoder.state_dict(), HyperParams.net_dir+HyperParams.net_name+HyperParams.net_run+'_decoder.pt')
        #     torch.save(model_dyn.state_dict(), HyperParams.net_dir+HyperParams.net_name+HyperParams.net_run+'_dyn.pt')
        if HyperParams.tolerance >= train_rmse:
            print('Early stopping!')
            break
        np.save(HyperParams.net_dir+'history'+HyperParams.net_run+'.npy', train_history)
        np.save(HyperParams.net_dir+'history_test'+HyperParams.net_run+'.npy', test_history)
    
    #print("\nLoading best network for epoch: ", best_epoch)
        torch.save(model_decoder.state_dict(), HyperParams.net_dir+HyperParams.net_name+HyperParams.net_run+'_decoder.pt')
        torch.save(model_dyn.state_dict(), HyperParams.net_dir+HyperParams.net_name+HyperParams.net_run+'_dyn.pt')
    model_decoder.load_state_dict(torch.load(HyperParams.net_dir+HyperParams.net_name+HyperParams.net_run+'_decoder.pt', map_location=torch.device('cpu')))
    model_dyn.load_state_dict(torch.load(HyperParams.net_dir+HyperParams.net_name+HyperParams.net_run+'_dyn.pt', map_location=torch.device('cpu')))

def get_data(dataloader, n):
    for i, data in enumerate(dataloader):
        if i == n:
            return data
    return None

def physics_loss(v_target, v_predicted, HyperParams):
    """
    Computes a physics-inspired loss between target and predicted vectors for each row and sums all terms.

    Parameters:
    - v_target (torch.Tensor): Target vectors.
    - v_predicted (torch.Tensor): Predicted vectors.
    - HyperParams (namespace): Hyperparameters (delta, epsilon).

    Returns:
    - torch.Tensor: Computed physics loss.

    Formula:
    loss = ||v_target - v_predicted||^2 / (2 - ||v_target||^2) + delta * ||(v_target / (epsilon + ||v_target||)) - (v_predicted / (epsilon + ||v_predicted||))||^2

    Example:
    ```python
    loss_value = physics_loss(target, predicted, hyperparameters)
    loss_value.backward()
    optimizer.step()
    ```
    """
    total_loss = 0

    for target, predicted in zip(v_target, v_predicted):
        norm_target = torch.norm(target)
        norm_predicted = torch.norm(predicted)

        term1 = torch.norm(target - predicted)**2 / (2 - norm_target**2)
        term2 = HyperParams.delta * torch.norm((target / (HyperParams.epsilon + \
                    norm_target)) - (predicted / (HyperParams.epsilon + norm_predicted)))**2

        total_loss += term1 + term2

    return total_loss/v_target.shape[0]