--> Import Following Libraries


import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
import urllib
import zipfile
from tqdm.auto import tqdm

--> Path
Change path values according to your folder path
data_dir = '/content/tiny-imagenet-200'

# Define training and validation data paths
train_dir = os.path.join(data_dir, 'train') 
test_dir = os.path.join(data_dir, 'test') 
val_dir = os.path.join(data_dir, 'val')

