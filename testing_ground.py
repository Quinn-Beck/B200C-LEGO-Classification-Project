import pandas as pd
import numpy as np
import torch
import glob

from preprocessing_pipeline import get_data

train_data, test_data, val_data = get_data(num_classes=2, include_val=True)

print(train_data.__len__(), test_data.__len__())


