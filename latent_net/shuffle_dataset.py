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



