import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import random
import utils
from data_distribution import data_distribution
import copy
from label_flipping_attack import poison_data, get_client_data_stat
import phases
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from model import LeNet
from arguments import load_arguments
from pathlib import Path
from tqdm import tqdm
import wandb

class trainFL:
    def __init__(self, args, global_network):
        self.args = args
        self.global_network = global_network
        self.CR_acc = []
        self.device_acc = []
        self.batch_size = args.batch_size
        self.lr = args.learning_rate
        self.niid = args.niid
        self.num_devices = args.client_num_in_total
        self.epochs = args.epochs
        self.c_rounds = args.comm_round
        self.num_malicious_devices = int(args.num_malicious_devices * args.client_num_in_total)
        self.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.device == "gpu" else "cpu")
        self.malicious_devices = random.sample(range(self.num_devices), self.num_malicious_devices)
        wandb.init(
            # set the wandb project where this run will be logged
            project="flproject",
            
            # track hyperparameters and run metadata
            config=vars(args)
        )        

        if (args.blur):
            self.train_dataset, self.test_dataset, self.train_dataset_with_blur = utils.data_loader(args)
        else:
            self.train_dataset, self.test_dataset = utils.data_loader(args)


        data_distribution(self.niid, self.train_dataset, self.num_devices)
        file_name = 'Distribution/mnist/data_split_niid_'+ str(self.niid)+ "_no_of_clients_" + str(self.num_devices)+'.pt'
        pt_file = torch.load(file_name)    
        seed = 123
        self.train_dataset_idxs = pt_file['datapoints']
        print("Malicious Devices: ", self.malicious_devices)
        self.g = torch.Generator()
        self.g.manual_seed(seed)    
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        
    def seed_worker(worker_id):
        np.random.seed(seed)
        random.seed(seed)

    def train(self):

        if (self.args.phase == 2):
            trusted_client = random.sample(range(self.num_devices), 1)
            while (trusted_client in self.malicious_devices):
                trusted_client = random.sample(range(self.num_devices), 1)
            print("Trusted Client:", trusted_client)
            dataset_to_train_global_model = torch.utils.data.Subset(self.train_dataset, self.train_dataset_idxs[trusted_client[0]])
        
        if (self.args.phase == 3):
            client_list = random.sample(range(self.num_devices), self.num_devices)
            set_malicious_devices = set(self.malicious_devices)
            trusted_clients =  [client for client in client_list if client not in set_malicious_devices]

            all_labels = get_client_data_stat(self.train_dataset)
            client_idx_dynamic_dataset = []
            dynamic_datasets = []
            client_labels = {}
            for i in trusted_clients:
                device_sample_labels = torch.utils.data.Subset(self.train_dataset, self.train_dataset_idxs[i])
                labels = get_client_data_stat(device_sample_labels)
                if (len(client_labels.keys() ^ labels.keys()) > 0):
                    client_idx_dynamic_dataset.append(i)
                    dynamic_datasets.append(device_sample_labels)
                    client_labels.update(labels)
                if (len(client_labels.items()) == len(all_labels.items())):
                    break
            

            print("Client Labels: ", client_labels.keys())
            print("Client Indexes to use: ", client_idx_dynamic_dataset)
            print("All Labels: ", all_labels.keys())


        to_df = []
        cosine_similarity_all_crounds = []
        for CR in tqdm(range(self.c_rounds), position=0, desc= "CR: "):
            #print('****************** CR ******************:',CR)
            local_weights = []
            for d in tqdm(range(self.num_devices), position=1):
                network = copy.deepcopy(self.global_network).to(self.device)
                optimizer = torch.optim.Adam(network.parameters(), lr=self.lr)
                device_sample = torch.utils.data.Subset(self.train_dataset, self.train_dataset_idxs[d])
                if (self.args.loss_function == "cross_entropy"):
                    criterion = F.cross_entropy           
                            
                if d in self.malicious_devices:
                    if (self.args.blur):
                        device_sample = torch.utils.data.Subset(self.train_dataset_with_blur, self.train_dataset_idxs[d])
                    if (self.args.label_flipping):
                        device_sample = poison_data(device_sample, self.args.no_of_labels_to_flip)
                    if (self.args.learning_rate_attack):
                        optimizer = torch.optim.Adam(network.parameters(), lr=self.args.learning_rate_poison_value)
                train_loader = DataLoader(dataset=device_sample, batch_size=self.batch_size, shuffle=True, worker_init_fn=self.seed_worker, generator=self.g)


                for epoch in range(self.epochs):
                    total_loss = 0

                    for _, (data, targets) in enumerate(train_loader):
                        data = data.to(device=self.device)
                        np.size(data)
                        targets = targets.to(device=self.device)
                        targets = targets.long()
                        network.train()
                        output = network(data) 
                        loss = criterion(output, targets)
                        total_loss += loss.item()

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                self_test_acc = utils.check_accuracy(DataLoader(dataset=device_sample,batch_size = self.batch_size),network,self.device)
                test_acc = utils.check_accuracy(DataLoader(dataset=self.test_dataset,batch_size = self.batch_size),network,self.device)
                wandb.log({'Test Accuracy':  {f'Client {d}:': round(test_acc,2)*100 }})
                wandb.log({'Training Loss': {f'Client {d}': total_loss/len(train_loader)}})
                wandb.log({'Self Test Acc': {f'Client {d}': round(self_test_acc,2)*100}})          
                
                local_weights.append(network.state_dict())
            if (self.args.phase == 1):
                to_df.append(phases.phase1(self.global_network, local_weights, self.device))
            elif (self.args.phase == 2):
                to_df.append(phases.phase2(self.global_network, local_weights, dataset_to_train_global_model, self.args, self.device))
            
            elif(self.args.phase == 3):
                to_df.append(phases.phase3(self.global_network, local_weights, dynamic_datasets, args=self.args, device=self.device))
            
            cosine_similarity_all_crounds = np.array(to_df)
            print()
            global_weights = utils.model_average(local_weights)
            self.global_network.load_state_dict(global_weights)
            global_test_acc = utils.check_accuracy(DataLoader(dataset=self.test_dataset, batch_size = self.batch_size), self.global_network,self.device)
            global_test_acc = round(global_test_acc*100,2)
            
            print(f'Global Test Acc {global_test_acc} %')

            self.CR_acc.append(global_test_acc)
            
            global_train_acc = utils.check_accuracy(DataLoader(dataset=self.train_dataset, batch_size = self.batch_size), self.global_network, self.device)
            wandb.log({"Global Network Main": {"test_acc": global_test_acc, "train_acc": global_train_acc}})
    
        filname = f'{self.args.model}_{self.args.dataset}_{self.args.client_num_in_total}clients_{self.args.num_malicious_devices}_niid{self.args.niid}_phase{self.args.phase}'

        Path(f"Results/{filname}").mkdir(parents=True, exist_ok=True)

        writer = pd.ExcelWriter(f'Results/{filname}/{filname}.xlsx', engine='xlsxwriter')


        to_df = np.array(cosine_similarity_all_crounds)

        print(to_df.shape)
        for i in range(0, to_df.shape[1]):
                df = pd.DataFrame(to_df[:, i, :])
                df.to_excel(writer, sheet_name='Dataset%d' % i)
                plt.clf()
                ax = sns.heatmap(df)
                ax.figure.savefig(f'Results/{filname}/{filname}_{i}_heatmap.png')
        writer.close()           

        plt.clf()  
        ax = sns.lineplot(y=self.CR_acc, x = range(len(self.CR_acc)))
        ax.figure.savefig(f'Results/{filname}/{filname}_test_accuracy_plot.png')


if __name__ == "__main__":
    args = load_arguments()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.device == "gpu" else "cpu")  
    global_network = LeNet(1).to(device)
    train_object = trainFL(args=args, global_network=global_network)
    train_object.train()