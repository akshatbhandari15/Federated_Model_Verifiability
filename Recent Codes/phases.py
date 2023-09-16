from collections import Counter
import copy
import random
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from tqdm import tqdm

import utils
import sampling


class FederatedModel:
    def __init__(
        self,
        device,
        num_devices=110,
        num_trusted_devices=None,
        use_trusted=False,
        num_malicious_devices=0,
        network='lenet',
        global_rounds=10,
        local_epochs=5,
        batch_size=64,
        learning_rate=0.001,
        optim='adam',
        dataset='mnist',
        split_type='iid',
        malicious_device_args={},
    ):
        self.network = network
        self.dataset = dataset
        self.num_devices = num_devices
        self.global_rounds = global_rounds
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.optim = optim
        self.device = device
        self.num_trusted_devices = num_trusted_devices
        self.use_trusted = use_trusted
        self.num_malicious_devices = num_malicious_devices
        self.malicious_device_args = malicious_device_args

        self.global_network = utils.get_network(network, dataset).to(device)
        self.train_dataset, self.test_dataset = utils.get_dataset(dataset)


        self.train_dataset_idxs = sampling.split_dataset(
            self.train_dataset, num_devices, split_type
        )

        while any(len(x) == 0 for x in self.train_dataset_idxs):
            self.train_dataset_idxs = sampling.split_dataset(
                self.train_dataset, num_devices, split_type
            )

        np.random.shuffle(self.train_dataset_idxs) # Needed?

        self.train_dataset_classes = [Counter([self.train_dataset[idx][1] for idx in idxs]) for idxs in self.train_dataset_idxs] #for every client which classes are present

        if self.num_trusted_devices is None and self.use_trusted:
            self.num_trusted_devices = 1
            self.trusted_devices = {0} #1st client is the trusted device
            self.trusted_devices_classes = self.train_dataset_classes[0]

            while len(self.trusted_devices_classes.keys()) < 10: # TODO: Don't hardcode classes
                x = random.choice(list(set(range(len(self.train_dataset_idxs))) - self.trusted_devices))
                if len(set(self.train_dataset_classes[x].keys()) - set(self.trusted_devices_classes.keys())) > 0:
                    self.trusted_devices |=  {x}
                    self.trusted_devices_classes += self.train_dataset_classes[x]
                    self.num_trusted_devices += 1
        
        self.train_dataset_idxs = (
            [x for i, x in enumerate(self.train_dataset_idxs) if i in self.trusted_devices] +
            [x for i, x in enumerate(self.train_dataset_idxs) if i not in self.trusted_devices]
        ) #1st trusted devices then rest all the devices 

        self.trusted_devices = list(range(self.num_trusted_devices))
        self.malicious_devices = list(range(self.num_trusted_devices, self.num_malicious_devices))

        if num_malicious_devices > 0 and malicious_device_args.get('wrong_dataset') == True:
            self.wrong_train_dataset, _ = utils.get_dataset('cifar10' if dataset == 'mnist' else 'mnist')

    def get_loader(self, dataset):
        return DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

    def get_local_dataset(self, d, is_malicious):
        if is_malicious:
            if self.malicious_device_args.get('flip') is not None:
                n_flip = self.malicious_device_args.get('flip')
                dataset = torch.utils.data.Subset(self.train_dataset, self.train_dataset_idxs[d])
                return utils.FlippedDataset(dataset, 10, n_flip) # TODO: Change 10
            elif self.malicious_device_args.get('wrong_dataset') == True:
                device_dataset_size = int(len(self.train_dataset) / self.num_devices)
                dataset_idxs = list(range(len(self.wrong_train_dataset)))
                random.shuffle(dataset_idxs)
                return torch.utils.data.Subset(
                    self.wrong_train_dataset, dataset_idxs[:device_dataset_size]
                )

        return torch.utils.data.Subset(self.train_dataset, self.train_dataset_idxs[d])

    def train(self, calculate_global_cosine=False, calculate_local_grad_cosine=False):
        stats = []
        for r in range(self.global_rounds):
            global_weights, _, round_stats = self.train_one_round(r, calculate_global_cosine, calculate_local_grad_cosine)

            self.global_network.load_state_dict(global_weights)
            round_stats['test_accuracy'] = utils.check_accuracy(
                self.get_loader(self.test_dataset),
                self.global_network,
                self.device
            )
            stats.append(round_stats)

        return pd.DataFrame(stats)
    
    def train_one_round(self, r, calculate_global_cosine=False, calculate_local_grad_cosine=False):
        malicious_devices = self.malicious_devices

        local_weights = []
        stats = {}

        for d in tqdm(range(self.num_devices)):
            local_weights.append(self.train_device(d, d in malicious_devices, stats))
            if calculate_global_cosine:
                local_network = utils.get_network(self.network, self.dataset).to(self.device)
                local_network.load_state_dict(local_weights[-1])
                stats[f'{d}_cosine'] = utils.model_cosine(self.global_network, local_network)

        if calculate_local_grad_cosine:
            for i in range(self.num_trusted_devices):
                for j in range(self.num_devices):
                    x = utils.get_network(self.network, self.dataset).to(self.device)
                    y = utils.get_network(self.network, self.dataset).to(self.device)
                    x.load_state_dict(local_weights[i])
                    y.load_state_dict(local_weights[j])
                    stats[f"{i}_{j}_cosine"] = utils.model_grad_cosine(
                        self.global_network, x, y
                    )
        
        global_weights = utils.model_average(local_weights)

        return global_weights, local_weights, stats

    def train_device(self, d, is_malicious, stats):
        network = copy.deepcopy(self.global_network).to(self.device)

        optim = self.get_optim(network, is_malicious)
        dataset = self.get_local_dataset(d, is_malicious)
        total_loss = 0

        for _, (data, targets) in enumerate(self.get_loader(dataset)):
            data = data.to(device=self.device)
            targets = targets.to(device=self.device)

            loss = F.cross_entropy(network(data), targets)
            total_loss += loss.item()

            optim.zero_grad()
            loss.backward()
            optim.step()

        return network.state_dict()

    def get_optim(self, model, is_malicious):
        lr = self.learning_rate
        if is_malicious and self.malicious_device_args.get('lr') is not None:
            lr = self.malicious_device_args['lr']

        if self.optim == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr)
        elif self.optim == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr)
        else:
            raise NotImplementedError("Invalid Optimizer")

def phase0(
    device,
    split_type,
    malicious_type = None,
    num_malicious_devices=0,
    fname='outputs/phase0',
    local_epochs=5, 
    global_rounds=10):

    plt.clf()
    f = None
    if malicious_type is None:
        f = f'{fname}_{split_type}_{num_malicious_devices}'
    else:
        mt = list(malicious_type.keys())[0]
        f = f'{fname}_{split_type}_{mt}_{malicious_type[mt]}_{num_malicious_devices}'

    model = FederatedModel(
        device=device, 
        split_type=split_type,
        malicious_device_args={} if malicious_type is None else malicious_type,
        num_malicious_devices=num_malicious_devices,
        local_epochs=local_epochs,
        global_rounds=global_rounds
        )

    stats = model.train().astype('float')
    stats.to_excel(f'{f}.xlsx')

    print(f)
    print(stats)


def phase1(
    device,
    split_type,
    malicious_type = None,
    num_malicious_devices=0,
    fname='outputs/phase1',
    local_epochs=5, 
    global_rounds=10):

    plt.clf()
    f = None
    if malicious_type is None:
        f = f'{fname}_{split_type}_{num_malicious_devices}'
    else:
        mt = list(malicious_type.keys())[0]
        f = f'{fname}_{split_type}_{mt}_{malicious_type[mt]}_{num_malicious_devices}'

    model = FederatedModel(
        device=device, 
        split_type=split_type,
        malicious_device_args={} if malicious_type is None else malicious_type,
        num_malicious_devices=num_malicious_devices,
        local_epochs=local_epochs,
        global_rounds=global_rounds
        )

    stats = model.train(calculate_global_cosine=True).astype('float')
    stats.to_excel(f'{f}.xlsx')

    plt.clf()
    ss = stats[[f'{i}_cosine' for i in range(110)]].astype(float).to_numpy()
    ax = sns.heatmap(ss)
    ax.figure.savefig(f'{f}_heatmap.png')

    print(f)
    print(stats)

def phase2(
    device,
    split_type,
    malicious_type = None,
    num_malicious_devices=0,
    fname='outputs/phase2',
    local_epochs=5, 
    global_rounds=10):

    plt.clf()
    f = None
    if malicious_type is None:
        f = f'{fname}_{split_type}_{num_malicious_devices}'
    else:
        mt = list(malicious_type.keys())[0]
        f = f'{fname}_{split_type}_{mt}_{malicious_type[mt]}_{num_malicious_devices}'

    model = FederatedModel(
        device=device, 
        split_type=split_type,
        malicious_device_args={} if malicious_type is None else malicious_type,
        num_malicious_devices=num_malicious_devices,
        num_trusted_devices=1,
        use_trusted=True,
        local_epochs=local_epochs,
        global_rounds=global_rounds
    )

    stats = model.train(calculate_local_grad_cosine=True).astype('float')
    stats.to_excel(f'{f}.xlsx')

    plt.clf()
    ss = stats[[f'0_{j}_cosine' for j in range(1, 110)]].astype(float).to_numpy()
    ax = sns.heatmap(ss)
    ax.figure.savefig(f'{f}_heatmap.png')

    print(f)
    print(stats)

def phase3(
    device,
    split_type,
    malicious_type = None,
    num_malicious_devices=0,
    fname='outputs/phase3',
    local_epochs=5, 
    global_rounds=10):

    plt.clf()
    f = None
    if malicious_type is None:
        f = f'{fname}_{split_type}_{num_malicious_devices}'
    else:
        mt = list(malicious_type.keys())[0]
        f = f'{fname}_{split_type}_{mt}_{malicious_type[mt]}_{num_malicious_devices}'

    model = FederatedModel(
        device=device, 
        split_type=split_type,
        malicious_device_args={} if malicious_type is None else malicious_type,
        num_malicious_devices=num_malicious_devices,
        num_trusted_devices=None, # Automatically assign trusted devices
        use_trusted=True,
        local_epochs=local_epochs,
        global_rounds=global_rounds
    )

    stats = model.train(calculate_local_grad_cosine=True).astype('float')
    stats.to_excel(f'{f}.xlsx')


    print(f)
    print(stats)

def experiment(device, niids, malicious_types, phase, global_rounds=10, local_epochs=5):
    phase(device=device, split_type='iid')

    for niid in niids:
        phase(
            device=device,
            split_type=f'niid-{niid}',
            local_epochs=local_epochs,
            global_rounds=global_rounds)

    for mt in malicious_types:
        phase(
            device=device, 
            split_type='iid', 
            malicious_type=mt,
            num_malicious_devices=30, 
            local_epochs = local_epochs,
            global_rounds=global_rounds)

        for niid in niids:
            phase(
                device=device, 
                split_type=f'niid-{niid}', 
                malicious_type=mt,
                num_malicious_devices=30, 
                local_epochs=local_epochs,
                global_rounds=global_rounds)