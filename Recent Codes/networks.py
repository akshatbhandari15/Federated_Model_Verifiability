import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()

        layers = [nn.Linear(in_dim, hidden[0]), nn.ReLU()]

        for x, y in zip(hidden, hidden[1:]):
            layers += [nn.Linear(x, y), nn.ReLU()]

        layers += [nn.Linear(hidden[-1], out_dim)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.model(x)


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = transforms.Resize((32, 32))(x)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
