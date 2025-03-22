import torch
import torch.optim as optim
import torch.nn as nn
from data import get_dataloaders
from model import LittleCNN
from tqdm import tqdm

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_model(epochs=10, learning_rate=0.001, batch_size=128, device='cuda'):

    return None

if __name__ == '__main__':
    train_model(epochs=10)
