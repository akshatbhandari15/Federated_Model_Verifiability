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
from label_flipping_attack import poison_data, get_client_data_stat
seed = 123
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
lr = 0.001
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
            
train_dataset = datasets.MNIST(root="./dataset", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./dataset", train=False, transform=transform, download=True)

def global_train_loop(network, dataset, epochs):
    print("#########################Global Model Training###############")
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0

        for _, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            np.size(data)
            targets = targets.to(device=device)
            targets = targets.long()
            network.train()
            output = network(data) 
            loss = F.cross_entropy(output, targets)
            total_loss += loss.item()

            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f'Epoch: {epoch+1}/{epochs} \tTraining Loss: {total_loss/len(train_loader):.6f}')
        self_test_acc = check_accuracy(DataLoader(dataset=dataset,batch_size = batch_size),network,device)
        print(f'Self Test Acc {round(self_test_acc,2)*100} %')
        
        test_acc = check_accuracy(DataLoader(dataset=test_dataset,batch_size = batch_size),network,device)
        print(f'Test Acc {round(test_acc,2)*100} %')
        print()



class LeNet(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
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


def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
                x = x.to(device=device)
                y = y.to(device=device)
                scores = model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum().item()
                num_samples += predictions.size(0)

    return (num_correct / num_samples)

with open('data_dist/other_data_niid_v3_30_10.txt') as f:
        train_dataset_idxs = json.loads(f.read())
    
#print(train_dataset_idxs.size())

num_devices = len(train_dataset_idxs)

num_malicious_devices = 20
malicious_devices = random.sample(range(len(train_dataset_idxs)), num_malicious_devices)
device_sample = torch.utils.data.Subset(train_dataset, train_dataset_idxs[0])

get_client_data_stat(device_sample)


def model_average(ws):
    n = len(ws)
    w_avg = copy.deepcopy(ws[0])
    for key in w_avg.keys():
        for i in range(1, n):
            w_avg[key] += ws[i][key]
        w_avg[key] = torch.div(w_avg[key], n)
    return w_avg

def seed_worker(worker_id):
    np.random.seed(seed)
    random.seed(seed)
    
g = torch.Generator()
g.manual_seed(seed)

