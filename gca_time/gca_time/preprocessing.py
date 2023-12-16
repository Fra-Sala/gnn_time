import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from gca_time import scaling


def graphs_dataset(dataset, HyperParams, params):
    """
    graphs_dataset: function to process and scale the input dataset for graph autoencoder model.

    Inputs:
    dataset: an object containing the dataset to be processed.
    HyperParams: an object containing the hyperparameters of the graph autoencoder model.

    Outputs:
    dataset_graph: an object containing the processed and scaled dataset.
    loader: a DataLoader object of the processed and scaled dataset.
    train_loader: a DataLoader object of the training set.
    test_loader: a DataLoader object of the test set.
    val_loader: a DataLoader object of the validation set.
    scaler_all: a scaler object to scale the entire dataset.
    scaler_test: a scaler object to scale the test set.
    xyz: an list containig array of the x, y and z-coordinate of the nodes.
    var: an array of the node features.
    VAR_all: an array of the scaled node features of the entire dataset.
    VAR_test: an array of the scaled node features of the test set.
    train_snapshots: a list of indices of the training set.
    test_snapshots: a list of indices of the test set.
    """

    xx = dataset.xx
    yy = dataset.yy
    xyz = [xx, yy]
    if dataset.dim == 3:
       zz = dataset.zz
       xyz.append(zz)
       
    var = torch.stack((dataset.VX, dataset.VY), dim=2)
    var = var.to(dtype=torch.float32)

    # PROCESSING DATASET
    num_nodes = var.shape[0]
    num_graphs = var.shape[1]

    print("Number of nodes processed: ", num_nodes)
    print("Number of graphs processed: ", num_graphs)
    #total_sims = int(num_graphs)
    #rate = HyperParams.rate/100
    #train_sims = int(rate * total_sims)
    #test_sims = total_sims - train_sims
    #main_loop = np.arange(total_sims).tolist()
    #np.random.shuffle(main_loop)
    #train_snapshots = main_loop[0:train_sims]
    #train_snapshots.sort()
    #test_snapshots = main_loop[train_sims:total_sims]
    #test_snapshots.sort()

    total_sims = len(params)
    rate = HyperParams.rate/100
    params_train = params[0:round(rate*total_sims)]
    params_test = params[round(rate*total_sims):total_sims]
    snapshots = list(range(num_graphs))
    train_sims = (num_graphs) // len(params)*len(params_train)
    test_sims = num_graphs - train_sims
    train_snapshots = snapshots[0:train_sims]
    test_snapshots = snapshots[train_sims:num_graphs]

    # SCALING DATASET
    var_test = var[:, test_snapshots, :]
    VAR_all, scaler_all = normalize_input(var)
    VAR_test, scaler_test = normalize_input(var_test)
    VAR_all = VAR_all.view(VAR_all.shape[0], VAR_all.shape[1], HyperParams.dim_sol).permute(1, 0, 2)
    VAR_test = VAR_test.view(VAR_test.shape[0], VAR_test.shape[1], HyperParams.dim_sol).permute(1, 0, 2)

    #scaling_type = HyperParams.scaling_type
    #scaler_all, VAR_all = scaling.tensor_scaling(var, scaling_type, HyperParams.scaler_number)
    #scaler_test, VAR_test = scaling.tensor_scaling(var_test, scaling_type, HyperParams.scaler_number)

    graphs = []
    edge_index = torch.t(dataset.E) - 1
    for graph in range(num_graphs):
        if dataset.dim == 2:
            pos = torch.cat((xx[:, graph].unsqueeze(1), yy[:, graph].unsqueeze(1)), 1)
        elif dataset.dim == 3:
            pos = torch.cat((xx[:, graph].unsqueeze(1), yy[:, graph].unsqueeze(1), zz[:, graph].unsqueeze(1)), 1)
        ei = torch.index_select(pos, 0, edge_index[0, :])
        ej = torch.index_select(pos, 0, edge_index[1, :])
        edge_diff = ej - ei
        if dataset.dim == 2:
            edge_attr = torch.sqrt(torch.pow(edge_diff[:, 0], 2) + torch.pow(edge_diff[:, 1], 2))
        elif dataset.dim == 3:
            edge_attr = torch.sqrt(torch.pow(edge_diff[:, 0], 2) + torch.pow(edge_diff[:, 1], 2) + torch.pow(edge_diff[:, 2], 2))
        node_features = VAR_all[graph, :]
        dataset_graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
        graphs.append(dataset_graph)

    HyperParams.num_nodes = dataset_graph.num_nodes
    train_dataset = [graphs[i] for i in train_snapshots]
    test_dataset = [graphs[i] for i in test_snapshots]

    print("Length of train dataset: ", len(train_dataset))
    print("Length of test dataset: ", len(test_dataset))

    loader = DataLoader(graphs, batch_size=1)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return loader, train_loader, test_loader, \
            val_loader, scaler_all, scaler_test, xyz, VAR_all, VAR_test, \
                train_snapshots, test_snapshots, params_train, params_test

def normalize_input(tensor):
    """ 
    Normalization according to the paper
    """
    alpha0 = (tensor.max(dim=0)[0] + tensor.min(dim=0)[0]) / 2
    alphaw = (tensor.max(dim=0)[0] - tensor.min(dim=0)[0]) / 2
    normalized_tensor = (tensor - alpha0) / alphaw
    return normalized_tensor, torch.stack((alpha0, alphaw))


# define the inverse function of normalize input
def inverse_normalize_input(normalized_tensor, scaler_all, snapshot_indx, HyperParams):
    # Select indices from alpha0 and alphaw using test_snapshot_indx
    alpha0, alphaw = scaler_all[0, :, :], scaler_all[1, :, :]
    alpha0_selected = alpha0[snapshot_indx, :]#.reshape(-1,1)
    alphaw_selected = alphaw[snapshot_indx, :]#.reshape(-1,1)

    # Reconstruct the tensor
    tensor = torch.zeros(normalized_tensor.shape)
    for j in range(HyperParams.dim_sol):
        tensor[:,j] = normalized_tensor[:,j] * alphaw_selected[j] + alpha0_selected[j]
       # tensor[i,:,0] = normalized_tensor[i,:,0] * alphaw_selected[i] + alpha0_selected[i]
    
    return tensor

