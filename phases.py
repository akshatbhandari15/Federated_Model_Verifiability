import torch.nn.functional as F
import torch
from utils import global_train_loop
import copy
from tqdm import tqdm

def phase1(global_network, local_weights, device):
    
    cosine_similarity = []
    flattened_weight_global = torch.Tensor([]).to(device=device)
    for key, value in global_network.state_dict().items():
        flattened_weight_global = torch.cat((flattened_weight_global, torch.flatten(value)))
    
    for i in range(len(local_weights)):
        flattened_weight = torch.Tensor([]).to(device=device)

        for key, value in local_weights[i].items():
            flattened_weight = torch.cat((flattened_weight, torch.flatten(value)))

        cosine_similarity.append(F.cosine_similarity(flattened_weight, flattened_weight_global, dim = 0).tolist())
    return cosine_similarity

def phase2(global_network, local_weights, dataset_to_train_global_model, args, device):
    #flattening global model
    gradient_weights = []
    cosine_similarity = []

    flattened_weight_global = torch.Tensor([]).to(device=device)
    for key, value in global_network.state_dict().items():
        flattened_weight_global = torch.cat((flattened_weight_global, torch.flatten(value)))
    #flattening local models
    for i in range(len(local_weights)):
        flattened_weight = torch.Tensor([]).to(device=device)

        for key, value in local_weights[i].items():
            #print(value)
            flattened_weight = torch.cat((flattened_weight, torch.flatten(value)))
 
        gradient_weights.append(torch.sub(flattened_weight, flattened_weight_global))
    
    #train global model    
    global_network_train = copy.deepcopy(global_network).to(device)
    global_train_loop(global_network_train, dataset_to_train_global_model, args, device=device, global_network_id=1)
    global_network_train_param = global_network_train.state_dict()
    ##flattening trained model
    flattened_weight_global_trained = torch.Tensor([]).to(device=device)
    for key, value in global_network_train_param.items():
        flattened_weight_global_trained = torch.cat((flattened_weight_global_trained, torch.flatten(value)))
    
    wc = torch.sub(flattened_weight_global_trained, flattened_weight_global)
    
    for i in range(len(gradient_weights)):
        cosine_similarity.append(F.cosine_similarity(gradient_weights[i], wc, dim = 0).tolist())
    return (cosine_similarity)


def phase3(global_network, local_weights, dynamic_datasets, args, device):
    #flattening global model
    cosine_similarity_array = []
    gradient_weights = []

    
    flattened_weight_global = torch.Tensor([]).to(device=device)
    for key, value in global_network.state_dict().items():
        flattened_weight_global = torch.cat((flattened_weight_global, torch.flatten(value)))
    #flattening local models
    for i in range(len(local_weights)):
        flattened_weight = torch.Tensor([]).to(device=device)
        for key, value in local_weights[i].items():
            flattened_weight = torch.cat((flattened_weight, torch.flatten(value)))
   
        gradient_weights.append(torch.sub(flattened_weight, flattened_weight_global))
    
    #train global model    
    for i in tqdm(range(len(dynamic_datasets)), desc = "Global Model Training: " , position=3):
        global_network_train = copy.deepcopy(global_network).to(device)
        cosine_similarity = []        
        global_train_loop(global_network_train, dynamic_datasets[i], args, device=device, global_network_id=i)
        global_network_train_param = global_network_train.state_dict()
    ##flattening trained model
        flattened_weight_global_trained = torch.Tensor([]).to(device=device)
        for key, value in global_network_train_param.items():
            flattened_weight_global_trained = torch.cat((flattened_weight_global_trained, torch.flatten(value)))
        
        wc = torch.sub(flattened_weight_global_trained, flattened_weight_global)
        
        for j in range(len(gradient_weights)):
            cosine_similarity.append(F.cosine_similarity(gradient_weights[j], wc, dim = 0).tolist())
        cosine_similarity_array.append(cosine_similarity)
    return cosine_similarity_array