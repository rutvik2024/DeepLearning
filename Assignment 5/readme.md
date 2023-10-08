# For both question import following libraries


from __future__ import print_function
#%matplotlib inline
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as td
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import DataLoader
from IPython.display import HTML
from torchvision.io import read_image
import torchvision.models as models
import torch.nn.init as init
import torch.nn.functional as F
import pickle
import cv2
import numpy as np
from PIL import Image
import dlib
import sys
from matplotlib import pyplot as plt
import dnnlib
import legacy
import PIL.Image
import numpy as np
import imageio
from tqdm.notebook import tqdm


# Then clone repository by using following command
git clone https://github.com/NVlabs/stylegan2-ada-pytorch

# To generate 10 random images execute following command
!python3 generate.py --outdir=out --trunc=1 --seeds=85,265,297,849,851,665,300,800,90,270 \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl


# Change path variable values in both the file. 