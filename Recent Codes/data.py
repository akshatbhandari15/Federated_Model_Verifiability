import numpy as np
from tqdm import tqdm
# import matplotlib.pyplot as plt
import math
import random
from collections import Counter
# import seaborn as sns
# from sklearn.utils import shuffle
# from util_functions import create_data,load_data,numpy_to_tensor
from torch.utils.data import Dataset, DataLoader
import random
import os
import torch

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# Return num_clients+1 data. The +1 is for the server test set to evaluate performance


def dist_data_per_client(num_clients, batch_size, non_iid, device):
    
    #write the code for data distribution
    
    
    return client_data_loaders, client_data_num_samples, test_loader


def plot(a,b):
    # pick a random client
    client = random.randint(a,b)
    # Get all the labels of this client
    tmp = client_data_labels[client]
    # Get the number of samples for each class for the current client
    tmp = Counter(tmp)
    # If a class does not exist, give it 0 samples
    class_counter = [tmp.get(i, 0) for i in classes]

    return client, class_counter
    
dist_data_per_client(200, 32, #betvalue_need_to_be_input, "cpu")


