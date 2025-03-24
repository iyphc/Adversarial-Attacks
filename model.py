import torch
import torch.nn as nn
import torch.nn.functional as F

class LittleCNN(nn.Module):
    def __init__(self):
        super(LittleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)  # 8x8 после двух MaxPool
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))      # 3x32x32 → 32x32x32
        x = F.max_pool2d(x, 2)         # 32x32x32 → 32x16x16
        x = F.relu(self.conv2(x))      # 32x16x16 → 64x16x16
        x = F.max_pool2d(x, 2)         # 64x16x16 → 64x8x8
        x = torch.flatten(x, 1)        # 64x8x8 → 4096
        x = F.relu(self.fc1(x))        # 4096 → 256
        x = self.fc2(x)                # 256 → 10
        return x
