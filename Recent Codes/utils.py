import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import copy
import random
import networks


def get_dataset(name):
    if name.lower() == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dataset = datasets.MNIST(
            root="./dataset/", train=True, transform=transform, download=True
        )

        test_dataset = datasets.MNIST(
            root="./dataset/", train=False, transform=transform, download=True
        )

        return train_dataset, test_dataset

    elif name.lower() == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        train_dataset = datasets.CIFAR10(
            root="./dataset/", train=True, download=True, transform=transform
        )

        test_dataset = datasets.CIFAR10(
            root="./dataset/", train=False, download=True, transform=transform
        )

        return train_dataset, test_dataset
    else:
        raise Exception("Invalid Dataset")

class FlippedDataset(Dataset):
    def __init__(self, dataset, n_classes, n_flip):
        self.dataset = dataset
        self.to_flip = np.random.choice(list(range(n_classes)), n_flip)
        self.flip = list(range(n_classes))
        for i in self.to_flip:
            self.flip[i] = (i + np.random.randint(1, 10)) % 10

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        #print("UTILS Flipped dataset x: ", x)
        #print("UTILS Flipped dataset y: ", self.flip[y])
        return x, self.flip[y]

def get_network(network, dataset):
    if dataset.lower() == "mnist":
        in_dim, num_classes = 28 * 28, 10
    elif dataset.lower() == "cifar10":
        in_dim, num_classes = 32 * 32 * 3, 10
    else:
        raise NotImplementedError("Dataset")

    if network.lower() == "mlp":
        network = networks.MLP(in_dim, num_classes, hidden=[512, 256, 128])
    elif network.lower() == "lenet":
        network = networks.LeNet()
    else:
        raise NotImplementedError("Model")

    return network


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
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    model.train()

    return (num_correct / num_samples).cpu().numpy()


def model_average(ws):
    n = len(ws)
    w_avg = copy.deepcopy(ws[0])
    for key in w_avg.keys():
        for i in range(1, n):
            w_avg[key] += ws[i][key]
        w_avg[key] = torch.div(w_avg[key], n)
    return w_avg


def model_sum(ws):
    n = len(ws)
    w_sum = copy.deepcopy(ws[0])
    for key in w_sum.keys():
        for i in range(1, n):
            w_sum[key] += ws[i][key]
    return w_sum


def model_divide(w, x):
    w = copy.deepcopy(w)
    for key in w.keys():
        w[key] /= x
    return w

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
        for x, y in zip(m.parameters(), n.parameters()):
            x, y = torch.flatten(x), torch.flatten(y)
            dot_m_n += torch.dot(x, y)
            dot_m_m += torch.dot(x, x)
            dot_n_n += torch.dot(y, y)

    return (dot_m_n / (torch.sqrt(dot_m_m) * torch.sqrt(dot_n_n))).cpu().numpy()


def model_norm(m, n, ord):
    norm = 0
    with torch.no_grad():
        for x, y in zip(m.parameters(), n.parameters()):
            x, y = torch.flatten(x), torch.flatten(y)
            norm += torch.norm(x - y, p=ord).cpu().numpy()
    return norm


def model_l1(m, n):
    return model_norm(m, n, 1)


def model_l2(m, n):
    return model_norm(m, n, 2)

def get_seperability(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        recall_score,
        precision_score
    ) 
    model = LogisticRegression(penalty='none')
    model.fit(X, y)
    y_pred = model.predict(X)
    return {
        'accuracy' : accuracy_score(y, y_pred),
        'precision' : precision_score(y, y_pred),
        'recall' : recall_score(y, y_pred),
    }