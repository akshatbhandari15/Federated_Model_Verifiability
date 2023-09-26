import torch.nn as nn
import torchvision.transforms as transforms
import torch    
import torch.nn.functional as F
from torchvision import models

class LeNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def create_model(args):
    if args.model == 'lenet':
        if args.dataset in ['mnist', 'fashionmnist', 'CUSTOM']:
            net = LeNet(in_channels=1, num_classes=10)
        elif args.dataset == 'cifar10':
            net = LeNet(in_channels=3, num_classes=10)
        else:
            net = LeNet(in_channels=3, num_classes=100)
    if args.model == 'resnet18':
        if args.dataset == 'cifar10':
            net = models.resnet18(num_classes=10)
        else:
            net = models.resnet18(num_classes=100)
    if args.model == 'resnet50':
        if args.dataset == 'cifar10':
            net = models.resnet50(num_classes=10)
        else:
            net = models.resnet50(num_classes=100)
    if args.model == 'vgg16':
        if args.dataset == 'cifar10':
            net = models.vgg16(num_classes=10)
        else:
            net = models.vgg16(num_classes=100)
    if args.model == 'AlexNet':
        if args.dataset == 'cifar10':
            net = models.alexnet(num_classes=10)
        else:
            net = models.alexnet(num_classes=100)
    if args.model == 'MobileNet':
        if args.dataset == 'cifar10':
            net = models.mobilenet_v3_small(num_classes=10)
        else:
            net = models.mobilenet_v3_small(num_classes=100)    
    if args.model == 'EfficientNet':
        if args.dataset == 'cifar10':
            net = models.efficientnet_v2_s(num_classes=10)
        else:
            net = models.efficientnet_v2_s(num_classes=100)
    return net
