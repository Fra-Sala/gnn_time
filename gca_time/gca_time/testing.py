import torch
from tqdm import tqdm
import numpy as np
from gca_time import gaussian_process


def evaluate(VAR, model_decoder, model_dyn, loader, params, times, HyperParams):
    """
    This function evaluates the trained model, returning predictions over the entire dataset (training+testing)
    and the evolution of the latent state s(t) for all simulations.
    Regarding the latent states,the idea is to assemble a tensor of dimensions
    [num_integration_times*num_simulations, latent state dim + 1], so that, for each simulation and for each 
    time for which an integration has been performed, we can retrieve the vector s(t) + time.
    This approach, that may seem not straightforward, was chosen aiming at the max flexibility, regardless of the 
    step of integration. It may be improved.
    
    Args:
        VAR (torch.Tensor): The variable tensor containing the entire dataset.
        model_decoder (Model): The decoder model.
        model_dyn (Model): The dynamic model.
        loader (DataLoader): The data loader.
        params (list): The parameters.
        times (list): The times.
        HyperParams (HyperParameters): The hyperparameters.
        
    Returns:
        results: tensor.
        latents: tensor.
    """

    counter = 0
    results = torch.zeros(VAR.shape[0], VAR.shape[1], 2)
    latents = []
    print("Evaluating the model...")
    for (param_indx, alpha) in enumerate(params):
        param_indx = counter//(len(times)-1)
        latents_states  = []
        stn_vec = []
        t_vec = []
        stn = torch.zeros(HyperParams.bottleneck_dim, device="cpu")
        latents_states.append(stn)
        t_vec.append(times[0])
        for (t_indx, t) in enumerate(times[1:]):
            num_integrations = round(float((times[t_indx+1])-times[t_indx])/HyperParams.dt)
            u_t = []
            for j in range(num_integrations):
                u_t = gaussian_process.eval_u_t(t_vec[-1], params[param_indx], HyperParams.T_f)
                u_t = torch.tensor(u_t, device="cpu")
                dyn_input = torch.cat((u_t.unsqueeze(0), stn), dim=0)
                stn_derivative = model_dyn(dyn_input)
                stn = stn + HyperParams.dt * stn_derivative
                t_vec.append(t_vec[-1] + HyperParams.dt)
                latents_states.append(stn)
                 
            data = get_data(loader, counter) #(param_indx*(len(times)-1)+t_indx)
            decoder_input = torch.cat((u_t.unsqueeze(0), stn), dim=0)
            results[counter, :, :] = model_decoder(decoder_input, data)
            counter += 1
        
        latents.append(torch.cat([torch.stack(latents_states), torch.tensor(t_vec).unsqueeze(1)], dim=1))
    print("Evaluation complete!")

    latents = torch.stack(latents)
    latents = latents.view(-1, latents.shape[-1])
    

    return results, latents


def get_data(dataloader, n):
    for i, data in enumerate(dataloader):
        if i == n:
            return data
    return None


                    
                    