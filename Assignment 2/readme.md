# For Question 1:

## You have to first run the following command

!pip install -U torch==1.7.1 torchtext==0.4.0

## Reload environment
exit()

# So that you can use functionality of Field and BucketIterator

# You have to import following libraries

import torch
import torchtext
from torchtext.data import Field, BucketIterator
import csv
import torch.nn as nn
from torchtext.datasets import TranslationDataset
from torchtext.data import TabularDataset
import random
import time
import matplotlib.pyplot as plt

# Now you have to mount with your google drive

from google.colab import drive
drive.mount('/content/drive')

# Now change following path according to your  drive path

train_path = '/content/drive/MyDrive/Dataset/Dakshina Dataset/hi/lexicons/hi.translit.sampled.train.tsv'
test_path = '/content/drive/MyDrive/Dataset/Dakshina Dataset/hi/lexicons/hi.translit.sampled.test.tsv'
val_path = '/content/drive/MyDrive/Dataset/Dakshina Dataset/hi/lexicons/hi.translit.sampled.dev.tsv'

# Thats it for question.


# For question 2

# import following libraries
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import io
import os
import zipfile
import random 
import math
import unicodedata
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# run the following code to get dataset
!wget 'http://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip'

!unzip 'household_power_consumption.zip'

# now change the path according to your file path
/content/household_power_consumption.txt
