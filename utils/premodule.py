import torch
import numpy as np
import os
import random

def Seed_Everything(seed):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device=='cuda':
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
#     device = 'cpu'
    print(device)
    return device