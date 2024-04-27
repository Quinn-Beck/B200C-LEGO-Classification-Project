import pandas as pd
import numpy as np
import torch
import glob
from tqdm import tqdm

from PIL import Image
from torchvision.transforms.v2 import Normalize
from torchvision.transforms.functional import to_tensor
from torch.utils.data import TensorDataset

# TODO: Get normalizer working and include to_tensor OR normalizer param for normalization
# across_classes = False -> use to_tensor
def get_data(num_classes=10, include_val=False):
    """
    Load processed images for model training and evaluation.
    
    Wraps all tensorized and scaled images from first ``num_classes`` classes in TensorDataset's
    to be used for model training and testing. Optionally returns validation data.
    
    Parameters
    ----------
    num_classes : int
        Number of classes present in the data.
    across_classes : bool, default False 
        Whether to normalize using class means and standard deviations.
    include_val : bool, default False
        Whether to return validation data.
        
    Returns
    -------
    train_data : TensorDataset
        TensorDataset containing labelled training images.
    test_data : TensorDataset
        TensorDataset containing labelled testing images.
    val_data : TensorDataset
        TensorDataset containing labelled validation images.
    """
    progress_bar = tqdm(range(num_classes*4000))
    # loading all folders and filenames into respective lists
    jpgfolders = glob.glob("C:/Users/Quinn/Desktop/B200C-Lego-Classification/64/*")
    jpgfiles = glob.glob("C:/Users/Quinn/Desktop/B200C-Lego-Classification/64/*/*.jpg")

    # define number of training, testing, and validation images per class
    if include_val == False:
        num_train = 3200
        num_test = 800
        num_val = 0
    
    else:
        num_train = 3000
        num_test = 500
        num_val = 500
    
    # iterate over first ``num_classes`` class folders
    for (iter, folder) in enumerate(jpgfolders):
        if iter > (num_classes - 1):
            break
        
        # initialize storage for imported images and their class labels
        raw_X = []
        raw_Y = torch.tensor([iter for k in range(num_train)])
        
        # iterate over image files
        for (idx, file) in enumerate(glob.glob(f"{folder}/*.jpg")):
            # to_tensor converts PIL image (H x W x C) to a torch.FloatTensor with values in the range [0.0, 1.0] 
            raw_X.append(to_tensor(Image.open(file)))
            progress_bar.update(1)
        
        # convert raw_X from a sequence of tensors to torch.tensor with batch dimension len(raw_X)
        raw_X = torch.stack(raw_X)
        
        # TODO: Normalize all images belonging to one class across channels
        # normalizer = Normalize(
        #                 mean = raw_X.mean(dim=[1,2]),
        #                 std = raw_X.std(dim=[1,2])
        #                 )
        
        # raw_X = normalizer(raw_X)
        
        # initialize train, test, and val sets
        if iter == 0:
            train_X = raw_X[0:num_train]
            test_X = raw_X[num_train:(4000-num_val)]
            val_X = raw_X[(4000-num_val):]
            train_Y = raw_Y
            test_Y = raw_Y[0:num_test]
            val_Y = raw_Y[0:num_val]
        
        # add the processed contents of current folder to existing train, test, and val sets    
        else:
            train_X = torch.cat((train_X, raw_X[0:num_train]), dim=0)
            test_X = torch.cat((test_X, raw_X[num_train:(4000-num_val)]), dim=0)
            val_X = torch.cat((val_X, raw_X[(4000-num_val):]), dim=0)
            train_Y = torch.cat((train_Y, raw_Y), dim=0)
            test_Y = torch.cat((test_Y, raw_Y[0:num_test]), dim=0)
            val_Y = torch.cat((val_Y, raw_Y[0:num_val]), dim=0)
            
    if include_val == False:
        # return train and test data zipped into TensorDatasets
        return TensorDataset(train_X, train_Y), TensorDataset(test_X, test_Y)
    else:
        # return train, test, and val data zipped into TensorDatasets
        return TensorDataset(train_X, train_Y), TensorDataset(test_X, test_Y), TensorDataset(val_X, val_Y)