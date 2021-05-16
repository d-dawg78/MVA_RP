import torch
from torch.utils.data import TensorDataset, DataLoader


def create_dataset(data, target):
    """
    Converts NumPy arrays into PyTorch datset.
    """    
    ds = TensorDataset(
        torch.tensor(data).float(), 
        torch.tensor(target).float())
    
    return ds


def create_3_part_dataset(raw, fft, target):
    """
    Converts NumPy arrays into PyTorch datset.
    """    
    ds = TensorDataset(
        torch.tensor(raw).float(), 
        torch.tensor(fft).float(), 
        torch.tensor(target).float())
    
    return ds


def create_4_part_dataset(raw, fft, wavelets, target):
    """
    Converts NumPy arrays into PyTorch datset.
    """    
    ds = TensorDataset(
        torch.tensor(raw).float(), 
        torch.tensor(fft).float(), 
        torch.tensor(wavelets).float(), 
        torch.tensor(target).float())
    
    return ds


def create_loader(data, bs=128, jobs=0):
    """
    Wraps the datasets returned by create_datasets function with data loader.
    """
    dl = DataLoader(data, batch_size=bs, shuffle=False, num_workers=jobs)

    return dl