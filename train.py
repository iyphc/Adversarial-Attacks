import torch
import torch.optim as optim
import torch.nn as nn
from data import get_dataloaders
from model import LittleCNN
from tqdm import tqdm
from evaluation import evaluate_model

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def train_model(epochs=10, learning_rate=0.001, batch_size=128, device='cuda'):
    device = get_device()
    train, test = get_dataloaders(batch_size=batch_size)
    size = len(train.dataset)
    model = LittleCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    i = 0
    for i in range(epochs):
        for batch, (X, y) in enumerate(train):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()

            # Backpropagation
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    test_accuracy = evaluate_model(model, test, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    torch.save(model.state_dict(), 'little_cnn.pth')
    print("Training complete. Model saved as little_cnn.pth")
    return model

if __name__ == '__main__':
    train_model(epochs=10)
