import torch
from torch.utils.data import DataLoader, Subset

def create_subset_dataloader(dataloader, num_samples, batch_size=128):
    indices = list(range(num_samples))
    subset = Subset(dataloader.dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
