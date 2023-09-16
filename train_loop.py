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
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.device == "gpu" else "cpu")
        self.malicious_devices = random.sample(range(self.num_devices), self.num_malicious_devices)
        self.trusted_client = random.sample(range(self.num_devices), 1)
        



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
        for CR in range(self.c_rounds):
            print('****************** CR ******************:',CR)
            
            local_weights = []
            cosine_similarity = []


            for d in range(self.num_devices):
                print('Device ID:',d)
                network = copy.deepcopy(self.global_network).to(self.device)
                optimizer = torch.optim.Adam(network.parameters(), lr=self.lr)
                #print(device_sample[0])
                #print(device_sample[0][0].shape)
                #train_dataset = datasets.MNIST(root="./dataset", train=True, transform=transform, download=True)
                device_sample = torch.utils.data.Subset(self.train_dataset, self.train_dataset_idxs[d])
                original_dev_sample = copy.deepcopy(device_sample)            
                            
                if d in self.malicious_devices:
                    if (self.args.blur):
                        device_sample = torch.utils.data.Subset(self.train_dataset_with_blur, self.train_dataset_idxs[d])
                    if (self.args.label_flipping):
                        device_sample = poison_data(device_sample, self.args.no_of_labels_to_flip)
                    if (self.args.learning_rate is not None):
                        optimizer = torch.optim.Adam(network.parameters(), lr=self.args.learning_rate_poison)


                    #print(device_sample[0][0].shape, device_sample[0][1].shape)
                
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
                        loss = F.cross_entropy(output, targets)
                        total_loss += loss.item()

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    print(f'Epoch: {epoch+1}/{self.epochs} \tTraining Loss: {total_loss/len(train_loader):.6f}')
                    self_test_acc = utils.check_accuracy(DataLoader(dataset=device_sample,batch_size = self.batch_size),network,self.device)
                    print(f'Self Test Acc {round(self_test_acc,2)*100} %')
                    
                    test_acc = utils.check_accuracy(DataLoader(dataset=self.test_dataset,batch_size = self.batch_size),network,self.device)
                    print(f'Test Acc {round(test_acc,2)*100} %')
                    print()
                
                local_weights.append(network.state_dict())


            to_df = []
            if (self.args.phase == 1):

                to_df.append(phases.phase1(self.global_network, local_weights, self.device))
            elif (self.args.phase == 2):
                trusted_client = random.sample(range(self.num_devices), 1)
                while (trusted_client in self.malicious_devices):
                    trusted_client = random.sample(range(self.num_devices), 1)
                print("Trusted Client:", trusted_client)
                dataset_to_train_global_model = torch.utils.data.Subset(self.train_dataset, self.train_dataset_idxs[trusted_client[0]])
                to_df.append(phases.phase2(self.global_network, local_weights, dataset_to_train_global_model, self.args, self.device))
            
            elif(self.args.phase == 3):
                client_list = random.sample(range(self.num_devices), self.num_devices)
                #print("Client List: ", client_list)
                set_malicious_devices = set(self.malicious_devices)
                trusted_clients =  [client for client in client_list if client not in set_malicious_devices]
                #print("Trusted_devices:", trusted_clients)

                all_labels = get_client_data_stat(self.train_dataset)
                client_idx_dynamic_dataset = []
                dynamic_datasets = []
                client_labels = {}
                for i in trusted_clients:
                    device_sample_labels = torch.utils.data.Subset(self.train_dataset, self.train_dataset_idxs[i])
                    labels = get_client_data_stat(device_sample_labels)
                    print("Client Labels: ", labels)
                    if (len(client_labels.keys() ^ labels.keys()) > 0):
                        client_idx_dynamic_dataset.append(i)
                        dynamic_datasets.append(device_sample_labels)
                        client_labels.update(labels)
                    if (len(client_labels.items()) == len(all_labels.items())):
                        break


                print("Client Labels: ", client_labels.keys())
                print("Client Indexes to use: ", client_idx_dynamic_dataset)
                print("All Labels: ", all_labels.keys())

                to_df.append(phases.phase3(self.global_network, local_weights, dynamic_datasets, args=self.args, device=self.device))
            
            
            filname = f'{self.args.model}_{self.args.dataset}_{self.args.client_num_in_total}clients_{self.args.num_malicious_devices}_niid{self.args.niid}_phase{self.args.phase}'
            #print(np.array(to_df.cpu()).shape)
            to_df = np.array(to_df)
            #print(to_df.shape)
            writer = pd.ExcelWriter(f'Results/{filname}.xlsx', engine='xlsxwriter')



            for i in range(0, to_df.shape[1]):
                #print(pd.DataFrame(to_df[:,:,i]))
                df = pd.DataFrame(to_df[:,i,:])
                df.to_excel(writer, sheet_name='Dataset%d' % i)
                
                plt.clf()
                #ss = stats[[f'{i}_{j}_cosine' for j in range(model.num_devices)]].astype(float).to_numpy()
                ax = sns.heatmap(df)
                ax.figure.savefig(f'Results/{filname}_{i}_heatmap.png')
            writer.close()           
            #df.to_excel(f'{self.args.filename}.xlsx')








            print()
        #     local_weights = [local_weights[i] for i in range(len(local_weights)) if i not in malicious_devices]
            global_weights = utils.model_average(local_weights)
            self.global_network.load_state_dict(global_weights)
            global_test_acc = utils.check_accuracy(DataLoader(dataset=self.test_dataset, batch_size = self.batch_size), self.global_network,self.device)
            global_test_acc = round(global_test_acc*100,2)
            
            print(f'Global Test Acc {global_test_acc} %')

            self.CR_acc.append(global_test_acc)
            
            dev_test_acc = utils.check_accuracy(DataLoader(dataset=device_sample, batch_size = self.batch_size),network, self.device)
            
            self.device_acc.append(dev_test_acc)