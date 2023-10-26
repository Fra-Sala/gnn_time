import torch
from torch import nn




class DynNet(torch.nn.Module):
    
        """ FCNN to learn the derivative wrt time of the latent state s(t)
        It has 2 hidden layers, and uses tanh as activation function"""
    
        def __init__(self, input_size, hidden_size, output_size):
            super().__init()
            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
            self.fc3 = torch.nn.Linear(hidden_size, output_size)
            self.activation = torch.nn.Tanh()
    
        def forward(self, x):
            x = self.activation(self.fc1(x))
            x = self.activation(self.fc2(x))
            x = self.fc3(x)
            return x
