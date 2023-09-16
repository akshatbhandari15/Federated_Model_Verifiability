    import random
from collections import Counter
import math

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def split_dataset(dataset, num_devices, split_type):
    if split_type.startswith("iid"):
        return split_dataset_iid(dataset, num_devices)
    else:
        p = int(split_type.split('-')[1])/100
        X = np.array([np.array(x) for x,_ in dataset])
        y = np.array([np.array(y) for _,y in dataset])
        y = y.reshape((y.shape[0], 1))
        return dist_data_per_client(X, y, num_devices, p)

def split_dataset_iid(dataset, num_devices):
    n = len(dataset)
    device_dataset_size = int(n / num_devices)
    dataset_idxs = list(range(n))
    random.shuffle(dataset_idxs)
    splits = []

    for i in range(0, n, device_dataset_size):
        splits.append(dataset_idxs[i: i+device_dataset_size])

    return splits

def dist_data_per_client(X_train, Y_train, num_clients, non_iid_per):
    classes = list(np.unique(Y_train))
    classes.sort()
    step = math.ceil(100/len(classes))
    min_client_in_chunk = 3 #70

    client_data_idxs = [list() for i in range(num_clients)]
    client_data_feats = [list() for i in range(num_clients)]
    client_data_labels = [list() for i in range(num_clients)]

    # This defines the amount of non-iid in terms of step size
    inter_non_iid_score = int((non_iid_per*100)/step)
    intra_non_iid_score = int((non_iid_per*100)%step)

    # This loop is used to find which chunk receives which classes
    class_chunks = list()
    tmp = list()
    for index, class_ in enumerate(classes):
        indices = np.arange(index,index+inter_non_iid_score)%len(classes)
        class_chunk = list(set(classes) - set(np.array(classes)[indices]))
        class_chunk.sort()
        class_chunks.append(class_chunk)
        tmp.extend(class_chunk)
    # val = Counter(tmp)[classes[0]]

    total_clients = num_clients
    clients_per_chunk = list()
    for i in range(len(class_chunks)):        
        clients_per_chunk.append(random.randint(min_client_in_chunk, total_clients - min_client_in_chunk*(len(class_chunks)-i-1)))
        total_clients-= clients_per_chunk[-1]
    cumulative_clients_per_chunk = [sum(clients_per_chunk[:i+1]) for i in range(len(clients_per_chunk))]

    class_count_dict = dict([[class_, 0] for class_ in classes])
    for index, class_chunk in enumerate(class_chunks):
        for class_label in class_chunk:
            indices = np.where(Y_train == class_label)[0]
            start = round(class_count_dict[class_label]*(len(indices)/Counter(tmp)[class_label]))
            end = round((class_count_dict[class_label]+1)*(len(indices)/Counter(tmp)[class_label]))
            class_count_dict[class_label]+=1
            indices = indices[start:end]
            num_data_per_client = math.ceil(len(indices)/clients_per_chunk[index])
            last_client_data = len(indices)%clients_per_chunk[index]

            val_last_client = 5
            x1, x2 = 1, clients_per_chunk[index]
            y1, y2 = num_data_per_client+last_client_data-val_last_client, val_last_client
            min_m, min_c = 0, val_last_client
            max_m = (y2-y1)/(x2-x1)
            max_c = y1-(max_m*x1)
            m = min_m + (((max_m - min_m)/(x2-x1))*intra_non_iid_score)
            c = min_c + (((max_c - min_c)/(x2-x1))*intra_non_iid_score)
            agg_points = 0

            denom = sum([m*(i+1) + c for i in range(clients_per_chunk[index])])
            weights = [(m*(i+1) + c)/denom for i in range(clients_per_chunk[index])]
            
            client_index_start = cumulative_clients_per_chunk[index-1] if index > 0 else 0
            client_index_end = cumulative_clients_per_chunk[index]
            for index_count, i in enumerate(np.arange(client_index_start, client_index_end)):
                if i >=num_clients:
                    break
                else:
                    # Each client gets a different number of samples as per the non-iid value
                    num_points = weights[index_count]*len(indices)
                    # Each client gets only a predfined number of samples
                    client_data_idxs[i].extend(indices[round(agg_points):round(agg_points+num_points)])
                    data = X_train[indices[round(agg_points):round(agg_points+num_points)]]
                    # Each client gets only a predfined number of labels of the current class
                    labels = [class_label for j in range(len(data))]
                    # Add the data to the client list
                    client_data_feats[i].extend(data)
                    # Add the labels to the client list
                    client_data_labels[i].extend(labels)
                    agg_points+= num_points
    return client_data_idxs

def visualize(dataset, device_idxs):
    vs = []
    for idxs in device_idxs:
        classes = Counter([dataset[idx][1] for idx in idxs])
        classes = sorted(classes.items(), key = lambda xc: xc[0])
        classes = [x for i,x in classes]
        vs.append(classes)
    vs = np.array(vs)
    print(vs.shape)
    vs = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(vs)
    plt.scatter(vs[:,0], vs[:,1], linewidth=0, s=10)
    plt.show()