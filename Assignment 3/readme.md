# For Question 1:

# You have to import following libraries

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
import pandas as pd
import numpy as np
from PIL import Image
from torchvision.models import resnet18
from collections import Counter, OrderedDict

# Now you have to mount with your google drive

from google.colab import drive
drive.mount('/content/drive')


# You have to upload dataset file on gdrive and then unzip and change path according to that
!unzip /content/drive/MyDrive/CelebA_Dataset.zip > /dev/null


# Thats it for question.