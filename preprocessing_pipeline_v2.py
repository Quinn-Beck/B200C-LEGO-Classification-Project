import random
import pandas as pd
import numpy as np
import torch
import glob

from PIL import Image
from torchvision.transforms.v2 import Normalize, ToTensor
from torch.utils.data import TensorDataset
from torchvision.datasets import ImageFolder

# TODO: ensure a 3200:400:400 train, val, test split for each class
# TODO: split files in class folder by 3200:400:400, concat then shuffle train, val, and test splits. Pull batches from these splits  

def data_generator(num_classes = 10):
    # get list of folder and image file paths
    jpgfolders = glob.glob("C:/Users/Quinn/Desktop/B200C-Lego-Classification/64/*")[0:num_classes]
    # does this return the correct images? Is it needed?
    jpgfiles = glob.glob("C:/Users/Quinn/Desktop/B200C-Lego-Classification/64/*/*.jpg")[0:num_classes*4000]

    label_dict = {f"{jpgfolders[i]}":(i) for i in range(len(jpgfolders))}
    
    train = []
    val = []
    test = []

    # get file paths for all images in all jpgfolders
    # and split into train, val, and test sets
    for folder in jpgfolders:
        images = glob.glob(f"{folder}/*.jpg")
        
        train += images[0:3200]
        val += images[3200:3600]
        test += images[3600:4000]

    
    # image paths are shuffled to ensure a random distribution of classes in each batch
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test

# yield batches of file paths taking into account end behaviour
def batcher(data, batch_size)
    assert len(data) // batch_size - 1 > 0
    n, r = divmod(len(data), batch_size)
    for i in range(n):
        yield data[i*batch_size:(i+1)*batch_size]
    yield data[-r:]

# utilizing Torch's ImageFolder class to load data!!!
# in the transforms argument we could pass a custom composition of transforms for on the fly data augmentation
ImageFolder(root = "C:/Users/Quinn/Desktop/B200C-Lego-Classification/64/*",
            transform = ToTensor,
            loader = Image
           )

# TODO: get equal representation of classes in the data splits, requires preslicing folders somehow
            
