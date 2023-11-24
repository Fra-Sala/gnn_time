import torch
from torch import nn
from gca_rom import scaling
import numpy as np

class HyperParams:
    """Class that holds the hyperparameters latent dyn-rec model.

    Args: ######### TO BE UPDATED ##############
        sparse_method (str): The method to use for sparsity constraint.
        rate (int): Amount of data used in training.
        seed (int): Seed for the random number generator.
        bottleneck_dim (int): The dimension of the bottleneck layer.
        tolerance (float): The tolerance value for stopping the training.
        lambda_map (float): The weight for the map loss.
        learning_rate (float): The learning rate for the optimizer.
        ffn (int): The number of feed-forward layers.
        in_channels (int): The number of input channels.
        hidden_channels (list): The number of hidden channels for each layer.
        act (function): The activation function to use.
        nodes (int): The number of nodes in each hidden layer.
        skip (int): The number of skipped connections.
        layer_vec (list): The structure of the network.
        net_name (str): The name of the network.
        scaler_name (str): The name of the scaler used for preprocessing.
        weight_decay (float): The weight decay for the optimizer.
        max_epochs (int): The maximum number of epochs to run training for.
        miles (list): The miles for learning rate update in scheduler.
        gamma (float): The gamma value for the optimizer.
        num_nodes (int): The number of nodes in the network.
        scaling_type (int): The type of scaling to use for preprocessing.
        net_dir (str): The directory to save the network in.
        cross_validation (bool): Whether to perform cross-validation.
    """

    def __init__(self, argv):

        self.net_name = argv[0]
        self.variable = argv[1]
        self.scaling_type = int(argv[2])
        self.scaler_number = int(argv[3])
        _, self.scaler_name = scaling.scaler_functions(self.scaler_number)
        #self.skip = int(argv[4])
        self.rate = int(argv[4])
        #self.sparse_method = 'L1_mean'
        #self.ffn = int(argv[6])
        #self.nodes = int(argv[7])
        #self.bottleneck_dim = int(argv[8])
        #self.lambda_map = float(argv[9])
        self.alpha_dyn = float(argv[5])
        self.alpha_rec = float(argv[6])
        self.dim_latent = int(argv[7])
        self.seed = 10
        self.tolerance = 1e-6
        self.learning_rate = 0.001
        #self.hidden_channels = [1]*self.in_channels
        #self.act = torch.tanh
        #self.layer_vec=[argv[11], self.nodes, self.nodes, self.nodes, self.nodes, self.bottleneck_dim]
        #self.net_run = '_' + self.scaler_name
        self.weight_decay = 0.00001
        self.dt = argv[8]
        self.max_epochs = argv[9]
        self.miles = []
        self.batch_pos_size =  15681 # number of positions per batch
        self.rff_encoded_mult = 2
        self.gamma = 0.0001
        #self.num_nodes = 0
        self.cross_validation = True
        self.num_pos_batches = 1 #number of batches (positions)
        self.T_f = 2.0
        self.net_dir = './' + 'latent_NN' + '/' + self.variable + '_' + self.net_name +  '_latdim' + str(self.dim_latent) \
                            + '_seed' + str(self.seed) + '_lr' + str(self.learning_rate) + '_sc' + str(self.scaling_type) + '_rate' + str(self.rate) + '/'



class DynNet(torch.nn.Module):

    """ FCNN to learn the derivative wrt time of the latent state s(t)
    It has 2 hidden layers, and uses tanh as activation function"""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)
        self.activation = torch.nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

            
class RecNet(torch.nn.Module):
    
    """ FCNN to learn the solution at a given time and position"""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)
        self.activation = torch.nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

