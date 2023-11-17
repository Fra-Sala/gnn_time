import torch
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
import numpy as np
from gca_rom import scaling
import rff


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




def process_and_scale_dataset(dataset, HyperParams):
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
    # PROCESSING DATASET
    num_nodes = var.shape[0]
    num_graphs = var.shape[1]

    print("Number of nodes processed: ", num_nodes)
    print("Number of shapshots processed: ", num_graphs)
    total_sims = int(num_graphs)
    rate = HyperParams.rate / 100
    train_sims = int(rate * total_sims)
    test_sims = total_sims - train_sims
    main_loop = list(range(total_sims))
    np.random.shuffle(main_loop)
    
    # Save indices of test and training snapshots
    train_snapshots = main_loop[0:train_sims]
    train_snapshots.sort()
    test_snapshots = main_loop[train_sims:total_sims]
    test_snapshots.sort()


    # FEATURE SCALING
    # save the velocity fields of both datasets (one column contains all the velocities
    var_train = dataset.U[:, train_snapshots]
    var_test = dataset.U[:, test_snapshots]

    # note that scaler is the type of scaling, while the second returned variable (e.g. VAR_all) is the original tensor, scaled
    scaling_type = HyperParams.scaling_type
    scaler_all, VAR_all = scaling.tensor_scaling(var, scaling_type, HyperParams.scaler_number)
    scaler_train, VAR_train = scaling.tensor_scaling(var_train, scaling_type, HyperParams.scaler_number)
    scaler_test, VAR_test = scaling.tensor_scaling(var_test, scaling_type, HyperParams.scaler_number)
 
    # VAR_all = var
    # VAR_train = var_train
    # VAR_test = var_test
    # scaler_all = 0
    # scaler_test = 0
    # Create PyTorch tensors for the scaled data (redundant, only to specify single precision)
    VAR_all_tensor = torch.tensor(VAR_all, dtype=torch.float64)
    VAR_train_tensor = torch.tensor(VAR_train, dtype=torch.float64)
    VAR_test_tensor = torch.tensor(VAR_test, dtype=torch.float64)

    # VAR_train_tensor = VAR_train_tensor.t()
    # VAR_test_tensor = VAR_test_tensor.t()

    # Create PyTorch DataLoader objects for training and testing data
    train_loader = DataLoader(VAR_train_tensor, batch_size=train_sims, shuffle=False)
    test_loader = DataLoader(VAR_test_tensor, batch_size=test_sims, shuffle=False)

     
   
    # Create the position dataset
    x_positions = dataset.xx[:, 0]
    y_positions = dataset.yy[:, 0]

    # Normalize the positions min-max
    # The positions are already inside the [0,1]^2, this normalization is redundant
    x_positions = (x_positions - x_positions.min()) / (x_positions.max() - x_positions.min())
    y_positions = (y_positions - y_positions.min()) / (y_positions.max() - y_positions.min())

    position_dataset = PositionDataset(x_positions, y_positions)

    


    return train_loader, test_loader, scaler_all, scaler_test, VAR_all, VAR_test, train_snapshots, test_snapshots, position_dataset
