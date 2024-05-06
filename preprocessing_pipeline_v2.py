import random
import pandas as pd
import numpy as np
import torch
import glob

from PIL import Image
from torchvision.transforms.v2 import Normalize
from torchvision.transforms.functional import to_tensor
from torch.utils.data import TensorDataset

# TODO: ensure a 3200:400:400 train, val, test split for each class
# TODO: split files in class folder by 3200:400:400, concat then shuffle train, val, and test splits. Pull batches from these splits  

def data_generator(num_classes = 10):
    # get list of folder and image file paths
    # TODO: determine if jpgfolders is needed
    # TODO: can we string slice the class label from the front of the file path?
    jpgfolders = glob.glob("C:/Users/Quinn/Desktop/B200C-Lego-Classification/64/*")[0:num_classes]
    jpgfiles = glob.glob("C:/Users/Quinn/Desktop/B200C-Lego-Classification/64/*/*.jpg")[0:num_classes*4000]

    for (iter, folder) in enumerate(jpgfolders):

    
    # image paths are shuffled to get a random distribution of classes in each batch
    random.shuffle(jpgfiles)

