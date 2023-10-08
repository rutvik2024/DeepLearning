You first have to install pytorch to run the code.
Here to run this code you have to import following libraries.

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as func
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import math

Other than that you have to download GuruMukhi dataset from gdrive then specify the train and validation path in train_data_path and val_data_path. then load dataset using ImageFolder().