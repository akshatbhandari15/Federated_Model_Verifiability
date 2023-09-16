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
seed = 123
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

batch_size = 64
lr = 0.001
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
            
train_dataset = datasets.MNIST(root="./dataset", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./dataset", train=False, transform=transform, download=True)


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

with open('data_dist/other_data_niid_0.txt') as f:
        train_dataset_idxs = json.loads(f.read())
    
#print(train_dataset_idxs.size())

num_devices = len(train_dataset_idxs)

num_malicious_devices = 20
malicious_devices = random.sample(range(len(train_dataset_idxs)), num_malicious_devices)

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


epochs = 2
c_rounds = 2
num_devices = len(train_dataset_idxs)

CR_acc = []
device_acc = []

global_network = LeNet(1).to(device)
print("Malicious Devices: ", malicious_devices)

#time.sleep(100)
for CR in range(c_rounds):
    print('****************** CR ******************:',CR)
    
    local_weights = []
    cosine_similarity = []

    for d in range(num_devices):
        print('Device ID:',d)
        network = copy.deepcopy(global_network).to(device)
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        device_sample = torch.utils.data.Subset(train_dataset, train_dataset_idxs[d])
        #print(device_sample[0])
        #print(device_sample[0][0].shape)

        original_dev_sample = copy.deepcopy(device_sample)
        if d in malicious_devices:
            device_sample = poison_data(device_sample)
            #print(device_sample[0][0].shape, device_sample[0][1].shape)
        else:
            optimizer = torch.optim.Adam(network.parameters(), lr=lr)
            pass
                                                
        #print(device_sample)
        #print(original_dev_sample)
        
        train_loader = DataLoader(dataset=device_sample, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)



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

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Epoch: {epoch+1}/{epochs} \tTraining Loss: {total_loss/len(train_loader):.6f}')
            self_test_acc = check_accuracy(DataLoader(dataset=original_dev_sample,batch_size = batch_size),network,device)
            print(f'Self Test Acc {round(self_test_acc,2)*100} %')
            
            test_acc = check_accuracy(DataLoader(dataset=test_dataset,batch_size = batch_size),network,device)
            print(f'Test Acc {round(test_acc,2)*100} %')
            print()


        
        local_weights.append(network.state_dict())

    flattened_weight_global = torch.Tensor([]).to(device=device)
    #print(local_weights[i])
    for key, value in global_network.state_dict().items():
        #print(value)
        flattened_weight_global = torch.cat((flattened_weight_global, torch.flatten(value)))
    
    for i in range(len(local_weights)):
        flattened_weight = torch.Tensor([]).to(device=device)

        #print(local_weights[i])
        for key, value in local_weights[i].items():
            #print(value)
            flattened_weight = torch.cat((flattened_weight, torch.flatten(value)))

        #print(flattened_weight)
        #print(flattened_weight_global)    
        cosine_similarity.append(F.cosine_similarity(flattened_weight, flattened_weight_global, dim = 0))
        
    print(cosine_similarity)


    print()
#     local_weights = [local_weights[i] for i in range(len(local_weights)) if i not in malicious_devices]
    global_weights = model_average(local_weights)
    global_network.load_state_dict(global_weights)
    global_test_acc = check_accuracy(DataLoader(dataset=test_dataset, batch_size = batch_size),global_network,device)
    global_test_acc = round(global_test_acc*100,2)
    
    print(f'Global Test Acc {global_test_acc} %')

    CR_acc.append(global_test_acc)
    
    dev_test_acc = check_accuracy(DataLoader(dataset=device_sample, batch_size = batch_size),network,device)
    
    device_acc.append(dev_test_acc)

x = range(c_rounds)
y = CR_acc

plt.plot(x, y, marker='o')
plt.xlabel('CR')
plt.ylabel('Accuracy')
plt.title('LR = 0.1')

plt.xticks(range(0, c_rounds, 1))
plt.yticks(range(10, 100, 10))
plt.ylim(10, 99)

plt.show()