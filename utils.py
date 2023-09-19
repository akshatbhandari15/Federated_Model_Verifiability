import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import copy
import random
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import wandb

def data_loader(args):

    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root="./dataset", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./dataset", train=False, transform=transform, download=True)

    if (args.blur):
        blur = transforms.GaussianBlur(args.kernal_size, sigma=args.sigma)
        transform_w_blur = transforms.Compose([transform, blur])
        train_dataset_with_blur = datasets.MNIST(root="./dataset", train=True, transform=transform_w_blur, download=True)
        return train_dataset, test_dataset, train_dataset_with_blur
    else:
        return train_dataset, test_dataset



def model_grad_cosine(m, a, b):
    dot_a_b, dot_a_a, dot_b_b = 0, 0, 0

    with torch.no_grad():
        for _m, _a, _b in zip(m.parameters(), a.parameters(), b.parameters()):
            _m = torch.flatten(_m)
            _a, _b = torch.flatten(_a), torch.flatten(_b)
            x = _a - _m
            y = _b - _m
            dot_a_b += torch.dot(x, y)
            dot_a_a += torch.dot(x, x)
            dot_b_b += torch.dot(y, y)

    return (dot_a_b / (torch.sqrt(dot_a_a) * torch.sqrt(dot_b_b))).cpu().numpy()


def model_cosine(m, n):
    dot_m_n, dot_m_m, dot_n_n = 0, 0, 0

    with torch.no_grad():
        for x, y in zip(m, n):
            x, y = torch.flatten(x), torch.flatten(y)
            dot_m_n += torch.dot(x, y)
            dot_m_m += torch.dot(x, x)
            dot_n_n += torch.dot(y, y)

    return (dot_m_n / (torch.sqrt(dot_m_m) * torch.sqrt(dot_n_n))).cpu().numpy()


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

def global_train_loop(network, dataset, args, device, global_network_id):
    g = torch.Generator()
    g.manual_seed(123)
    #print("#########################Global Model Training###############")
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
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


        #print(f'Epoch: {epoch+1}/{args.epochs} \tTraining Loss: {total_loss/len(train_loader):.6f}')
        self_test_acc = check_accuracy(DataLoader(dataset=dataset , batch_size = args.batch_size), network , device)
        #print(f'Self Test Acc {round(self_test_acc,2)*100} %')
        wandb.log({'Global Network {global_network_id}': {'Training Loss': total_loss/len(train_loader)}})
        wandb.log({'Global Network {global_network_id}': {'Self Test Acc': round(self_test_acc,2)*100}})
