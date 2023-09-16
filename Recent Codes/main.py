import torch
from phases import *
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
niids = [20, 40, 60, 80]
malicious_types = [{'flip': 5}, {'wrong_dataset': True}, {'lr':0.5}, {'lr':0.00001}]

experiment(device, niids, malicious_types, phase3)
# experiment(device, niids, malicious_types, phase1)
# experiment(device, niids, malicious_types, phase2)
# experiment_phase1(device, niids, malicious_types)
# experiment_phase2(device, niids, malicious_types, num_trusted_devices=1)
# experiment_phase2(device, niids, malicious_types, num_trusted_devices=3)
exit()