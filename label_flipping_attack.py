from numpy import random
import torch
from torch.utils.data import TensorDataset

def replace_original_class_with_target_class(
        data_labels, no_of_labels_to_flip
):
    """
    :param targets: Target class IDs
    :type targets: list
    :return: new class IDs
    """


    data_labels_set = set(data_labels)
    
    if( no_of_labels_to_flip > len(data_labels_set)):
        no_of_labels_to_flip = len(data_labels_set)
    original_class_list = random.choice(data_labels_set, no_of_labels_to_flip, replace = False)
    target_class_list = random.choice(data_labels_set, no_of_labels_to_flip, replace = False)
    
    for i in range(len(target_class_list)):
        while (original_class_list[i] == target_class_list[i]):
            target_class_list[i] = random.choice(data_labels_set)


    if (
            len(original_class_list) == 0
            or len(target_class_list) == 0
            or original_class_list is None
            or target_class_list is None
    ):
        return data_labels
    if len(original_class_list) != len(target_class_list):
        raise ValueError(
            "the length of the original class list is not equal to the length of the targeted class list"
        )


    for i in range(len(original_class_list)):
        for idx in range(len(data_labels)):
            if data_labels[idx] == original_class_list[i]:
                data_labels[idx] = target_class_list[i]
    return data_labels

def get_client_data_stat(local_dataset):
    #print("-==========================")
    targets_set = {}
    for batch_idx, (data, targets) in enumerate(local_dataset):
        for t in [targets]:
            if t in targets_set.keys():
                targets_set[t] += 1
            else:
                targets_set[t] = 1

    total_counter = 0

    for item in targets_set.items():
        total_counter += item[1]
    return targets_set



def poison_data(local_dataset, no_of_labels_to_flip):
    get_client_data_stat(local_dataset)

    tmp_local_dataset_x = torch.Tensor([])
    tmp_local_dataset_y = torch.Tensor([])
    targets_set = {}
    for batch_idx, (data, targets) in enumerate(local_dataset):
        targets =torch.Tensor([targets])
        tmp_local_dataset_x = torch.cat((tmp_local_dataset_x, data))
        tmp_local_dataset_y = torch.cat((tmp_local_dataset_y, targets))

        for t in targets.tolist():
            if t in targets_set.keys():
                targets_set[t] += 1
            else:
                targets_set[t] = 1
    total_counter = 0
    for item in targets_set.items():
        total_counter += item[1]
    

    tmp_y = replace_original_class_with_target_class(
        data_labels=tmp_local_dataset_y, no_of_labels_to_flip=no_of_labels_to_flip
    )
    tmp_local_dataset_x = torch.unsqueeze(tmp_local_dataset_x, 1)
    dataset = TensorDataset(tmp_local_dataset_x, tmp_y)
    #poisoned_data = DataLoader(dataset, batch_size=64)
    #get_client_data_stat(poisoned_data)

    return dataset