import torch.nn as nn
import torch.nn.functional as F

class LittleCNN(nn.Module): # Наследуемся от nn.Module, чтобы применять методы тензорфлов для класса
    def __init__(self, num_classes=10):
        super(LittleCNN, self).__init__()
        # Сверточные слои
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Полносвязные слои
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 8, 8]
        x = x.view(x.size(0), -1)             # Преобразуем в вектор
        x = F.relu(self.fc1(x))               # Перевод в полносвязный слой с ReLU
        # Выходной слой, softmax не применяется, т.к. CrossEntropyLoss включает log_softmax
        x = self.fc2(x)
        return x