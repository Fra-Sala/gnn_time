import torch
from tqdm import tqdm
import numpy as np
from gca_time import gaussian_process


def evaluate(VAR, model_decoder, model_dyn, loader, params, times, HyperParams):
    """
    This function evaluates the performance of a trained Autoencoder (AE) model.
    It encodes the input data using both the model's encoder and a mapping function,
    and decodes the resulting latent representations to obtain predicted solutions.
    The relative error between the two latent representations is also computed.

    Inputs:
    VAR: np.array, ground truth solution
    model: object, trained AE model
    loader: object, data loader for the input data
    params: np.array, model parameters
    HyperParams: class, model architecture and training parameters

    Returns:
    results: np.array, predicted solutions
    latents_map: np.array, latent representations obtained using the mapping function
    latents_gca: np.array, latent representations obtained using the AE encoder
    """

    # results = torch.zeros(VAR.shape[0], VAR.shape[1], 1)
    # latents_map = torch.zeros(VAR.shape[0], HyperParams.bottleneck_dim)
    # latents_gca = torch.zeros(VAR.shape[0], HyperParams.bottleneck_dim)
    # index = 0
    # latents_error = list()
    # with torch.no_grad():
    #     for data in tqdm(loader):
    #         z_net = model.solo_encoder(data)
    #         z_map = model.mapping(params[test[index], :])
    #         latents_map[index, :] = z_map
    #         latents_gca[index, :] = z_net
    #         lat_err = np.linalg.norm(z_net - z_map)/np.linalg.norm(z_net)
    #         latents_error.append(lat_err)
    #         results[index, :, :] = model.solo_decoder(z_map, data)
    #         index += 1
    #     np.savetxt(HyperParams.net_dir+'latents'+HyperParams.net_run+'.csv', latents_map.detach(), delimiter =',')
    #     latents_error = np.array(latents_error)
    #     # print("\nMaximum relative error for latent  = ", max(latents_error))
    #     # print("Mean relative error for latent = ", sum(latents_error)/len(latents_error))
    #     # print("Minimum relative error for latent = ", min(latents_error))
    # return results, latents_map, latents_gca

    #for (param_indx, alpha) in enumerate(params_test):
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
        # Go through the decoder
        #data = data.to(device)
            
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


                    
                    