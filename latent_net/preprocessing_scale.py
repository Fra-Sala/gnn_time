import torch
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
import numpy as np
from gca_rom import scaling


class PositionDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_pos = self.x[idx]
        y_pos = self.y[idx]
        return (x_pos, y_pos)




def process_and_scale_dataset(dataset, HyperParams, params):
    """
    process_and_scale_dataset: function to process and scale the input dataset.

    Inputs:
    dataset: an object containing the dataset to be processed.
    HyperParams: an object containing the hyperparameters of the model.

    Outputs:
    train_loader: a DataLoader object of the training set.
    test_loader: a DataLoader object of the test set.
    scaler_all: a scaler object to scale the entire dataset.
    scaler_test: a scaler object to scale the test set.
    VAR_all: an array of the scaled node features of the entire dataset.
    VAR_test: an array of the scaled node features of the test set.
    train_snapshots: a list of indices of the training set.
    test_snapshots: a list of indices of the test set.
    """

    # Extract data from the dataset object
    var = dataset.U
    total_sims = len(params)
    # PROCESSING DATASET
    num_nodes = var.shape[0]
    num_graphs = var.shape[1]//int(HyperParams.dim)

    print("Number of nodes processed: ", num_nodes)
    print("Number of simulations processed: ", total_sims)

    # Only one snapshot, same for test and train
    # train_snapshots = [0]
    # test_snapshots = [0]
    # params_train = params
    # params_test = params
    # train_sims = 1
    # test_sims = 1

    # train_snapshots = np.array([0])
    # test_snapshots = np.array([1])
    # params_train = params[0]
  
    # params_test = params[1]

    # train_sims = 6
    # test_sims = 6



    rate = HyperParams.rate / 100
    # Split params in two vectors, params_train and params_test, according to the rate
    params_train = params[0:round(rate*total_sims)]
    params_test = params[round(rate*total_sims):total_sims]
    # Retrieve the number of snapshots per single parameter
    num_snapshots = int(num_graphs/total_sims)
    # Create a list of indices of the snapshots
    snapshots = list(range(num_graphs))
    # Take the train_snapshots associated with params_train
    train_sims = (num_graphs) // len(params)*len(params_train)
    test_sims = num_graphs - train_sims
    train_snapshots = snapshots[0:train_sims]
    # Take the test_snapshots associeted with params_test
    test_snapshots = snapshots[train_sims:num_graphs]

  

    # FEATURE SCALING
    # save the velocity fields of both datasets (one column contains all the velocities
    if HyperParams.dim == 1:
        train_indices = train_snapshots
        test_indices = test_snapshots
    else:
        train_indices = [i for snapshot in train_snapshots for i in range(snapshot*HyperParams.dim, (snapshot+1)*HyperParams.dim)]
        test_indices = [i for snapshot in test_snapshots for i in range(snapshot*HyperParams.dim, (snapshot+1)*HyperParams.dim)]

    var_train = torch.tensor(dataset.U[:, train_indices])
    var_test = torch.tensor(dataset.U[:, test_indices])


    VAR_all, scaler_all = normalize_input(torch.tensor(var))
    VAR_train,scaler_train = normalize_input(var_train)
    VAR_test, scaler_test = normalize_input(var_test)

    # Note that scaler is the type of scaling, while the second returned variable (e.g. VAR_all) is the original tensor, scaled
    # scaling_type = HyperParams.scaling_type
    # scaler_all, VAR_all = scaling.tensor_scaling(var, scaling_type, HyperParams.scaler_number)
    # scaler_train, VAR_train = scaling.tensor_scaling(var_train, scaling_type, HyperParams.scaler_number)
    # scaler_test, VAR_test = scaling.tensor_scaling(var_test, scaling_type, HyperParams.scaler_number)

    # Create PyTorch tensors for the scaled data (redundant, only to specify single precision)
    VAR_all_tensor = torch.tensor(VAR_all, dtype=torch.float64)
    VAR_train_tensor = torch.tensor(VAR_train, dtype=torch.float64)
    VAR_test_tensor = torch.tensor(VAR_test, dtype=torch.float64)

    # Reshape the tensors to add a third dimension in case of multiple components of velocity 
    n_pos = VAR_all_tensor.shape[0]
    VAR_all_tensor = VAR_all_tensor.view(n_pos, VAR_all_tensor.shape[1] // HyperParams.dim, HyperParams.dim).permute(1, 0, 2)
    VAR_train_tensor = VAR_train_tensor.view(n_pos, VAR_train_tensor.shape[1] // HyperParams.dim, HyperParams.dim).permute(1, 0, 2)
    VAR_test_tensor = VAR_test_tensor.view(n_pos, VAR_test_tensor.shape[1] // HyperParams.dim, HyperParams.dim).permute(1, 0, 2)



    # VAR_all_tensor = VAR_all_tensor.permute(1, 0)
    # VAR_train_tensor = VAR_train_tensor.permute(1, 0)
    # VAR_test_tensor = VAR_test_tensor.permute(1, 0)
    # # Unsqueeze to be consistent with the dimension in gca_rom. Possibly to be modified
    # VAR_all_tensor = VAR_all_tensor.unsqueeze(2)
    # VAR_train_tensor = VAR_train_tensor.unsqueeze(2)
    # VAR_test_tensor = VAR_test_tensor.unsqueeze(2)



    # Create PyTorch DataLoader objects for training and testing data
    train_loader = DataLoader(VAR_train_tensor, batch_size=train_sims*HyperParams.dim, shuffle=False)
    test_loader = DataLoader(VAR_test_tensor, batch_size=test_sims*HyperParams.dim, shuffle=False)
    
    # Create the position dataset
    x_positions = dataset.xx[:, 0]
    y_positions = dataset.yy[:, 0]
    # Normalize the positions using normalize_input
    x_positions, _ = normalize_input(x_positions)
    y_positions, _ = normalize_input(y_positions)

    #position_dataset = PositionDataset(x_positions, y_positions)
    # Create position_dataset by putting x_positions and y_positions next to each other
    x_positions = x_positions.unsqueeze(1)
    y_positions = y_positions.unsqueeze(1)
    position_dataset = torch.cat((x_positions,y_positions), dim=1)

    


    return train_loader, test_loader, scaler_all, scaler_test, VAR_all, VAR_test,\
             train_snapshots, test_snapshots, position_dataset, params_train, params_test



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
    alpha0, alphaw = scaler_all[0], scaler_all[1]
    alpha0_selected = alpha0[snapshot_indx*HyperParams.dim:snapshot_indx*HyperParams.dim+HyperParams.dim]#.reshape(-1,1)
    alphaw_selected = alphaw[snapshot_indx*HyperParams.dim:snapshot_indx*HyperParams.dim+HyperParams.dim]#.reshape(-1,1)

    # Reconstruct the tensor
    tensor = torch.zeros(normalized_tensor.shape)

    for j in range(HyperParams.dim):
        tensor[:,j] = normalized_tensor[:,j] * alphaw_selected[j] + alpha0_selected[j]
       # tensor[i,:,0] = normalized_tensor[i,:,0] * alphaw_selected[i] + alpha0_selected[i]
    
    return tensor
