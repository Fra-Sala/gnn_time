import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data.sampler import BatchSampler, SequentialSampler

def shuffle_position_dataset(position_dataset, HyperParams, seed):

    np.random.seed(seed)
    # Shuffle the position batches in a "controlled" way
    # Create a list of batch indices and shuffle it
    num_batches = len(position_dataset) // HyperParams.batch_size_pos
    batch_indices = np.arange(num_batches)
    np.random.shuffle(batch_indices)

    # Use the shuffled batch indices to index the DataLoader
    sampler = BatchSampler(SequentialSampler(range(len(position_dataset))), batch_size=HyperParams.batch_size_pos, drop_last=False)
    pos_batch_sampler = [list(sampler)[i] for i in batch_indices]
    position_loader = DataLoader(position_dataset, batch_sampler=pos_batch_sampler)

    return pos_batch_sampler, position_loader


# Mock position_dataset
position_dataset = np.arange(10)

# Mock HyperParams
class HyperParams:
    batch_size_pos = 2

# Mock seed
seed = 42

# Call the function
pos_batch_sampler, position_loader = shuffle_position_dataset(position_dataset, HyperParams, seed)

print(pos_batch_sampler, position_loader)

u = np.array([25,23,45,56,46,45,21,33,1,2])
print(u[np.array(pos_batch_sampler[0])])


# # Your data
# data = np.arange(10)
# dataset = MyDataset(data)

# # Define your batch size
# batch_size = 2

# # Create a list of batch indices and shuffle it
# num_batches = len(dataset) // batch_size
# batch_indices = np.arange(num_batches)
# np.random.shuffle(batch_indices)

# # Use the shuffled batch indices to index the DataLoader
# sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(range(len(dataset))), batch_size=batch_size, drop_last=False)
# batch_sampler = [list(sampler)[i] for i in batch_indices]

# data_loader = DataLoader(dataset, batch_sampler=batch_sampler)

# # Now, you can iterate over data_loader to get your batches in the shuffled order
# for data in data_loader:
#     print(data)

# # And you can keep the 'batch_indices' array for future reference
# print("Shuffle order:", batch_indices)

# # Your second vector
# velocity_vec = np.arange(10, 20) 

# print(batch_sampler)

# # Access batches of the second vector
# for batch_indices in batch_sampler:
#     ten_vel = velocity_vec[batch_indices]
#     print(ten_vel)