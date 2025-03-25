import torch
import torch.nn as nn
from data import get_dataloaders
from model import LittleCNN
from tqdm import tqdm
from utils import get_device
from evaluation import evaluate_model

def train_model(epochs=10, learning_rate=0.001, batch_size=128):
    device = get_device()
    train, test = get_dataloaders(batch_size=batch_size)
    size = len(train.dataset)
    model = LittleCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(epochs):
        model.train()
        running_loss = 0
        total = 0

        for X, y in tqdm(train, desc=f'Epoch {i+1}/{epochs}'):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X.size(0)
            total += y.size(0)
        
        epoch_loss = running_loss / total
        print(f"Epoch {i+1}/{epochs} - Loss: {epoch_loss:.4f}")

    test_accuracy = evaluate_model(model, test, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    torch.save(model.state_dict(), 'little_cnn.pth')
    print("Training complete. Model saved as little_cnn.pth")
    return model

if __name__ == '__main__':
    train_model(epochs=10, batch_size=256)
