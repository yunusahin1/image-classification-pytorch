import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 5) # (12, 252, 252)
        self.pool = nn.MaxPool2d(2, 2) # (12, 126, 126)
        self.conv2 = nn.Conv2d(12, 24, 5) # (24, 122, 122) (24, 61, 61)
        self.fc1 = nn.Linear(24 * 61 * 61, 120) # flatten (24 * 61 * 61)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
