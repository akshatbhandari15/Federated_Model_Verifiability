from arguments import load_arguments
from data_distribution import data_distribution
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import time
import json
import copy
import numpy as np
import random
import matplotlib.pyplot as plt
from label_flipping_attack import poison_data
import utils
from model import LeNet
from train_loop import trainFL

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
args = load_arguments()
device = torch.device("cuda" if torch.cuda.is_available() and args.device == "gpu" else "cpu")  
global_network = LeNet(1).to(device)
train_object = trainFL(args=args, global_network=global_network)
train_object.train()

"""
x = range(c_rounds)
y = CR_acc

plt.plot(x, y, marker='o')
plt.xlabel('CR')
plt.ylabel('Accuracy')
plt.title('LR = 0.1')

plt.xticks(range(0, c_rounds, 1))
plt.yticks(range(10, 100, 10))
plt.ylim(10, 99)

plt.show()"""