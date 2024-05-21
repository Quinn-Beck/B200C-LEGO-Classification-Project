import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from prettytable import PrettyTable
from torchvision.models import mobilenet_v3_small
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset
from tools.preprocessing_pipeline_v2 import data_generator
from tools.fit import train, validate

# --- INSTANTIATE / LOAD MODEL ---
mobilenet_model = mobilenet_v3_small(weights='DEFAULT')
# get number of in features from source
num_features = mobilenet_model.classifier[3].in_features
# redefine the networks final fully connected layer
mobilenet_model.classifier[3] = nn.Linear(num_features, num_classes)
# send to gpu
mobilenet_model = mobilenet_model.to(device)

# --- TRAINING ---
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=mobilenet_model.parameters(), lr=learn_rate)

# initialize training/val loss & accuracy
mobile_train_loss = []
mobile_train_acc = []
mobile_val_loss = []
mobile_val_acc = []

table = PrettyTable()
table.field_names = ['Epoch', 'Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy']

for epoch in range(num_epochs):
    # training loop
    train_epoch_loss, train_epoch_acc = train(mobilenet_model, train_dl, optimizer, loss_fn)
    mobile_train_loss.append(train_epoch_loss)
    mobile_train_acc.append(train_epoch_acc)
    # validation loop
    val_epoch_loss, val_epoch_acc = validate(mobilenet_model, val_dl, loss_fn)
    mobile_val_loss.append(val_epoch_loss)
    mobile_val_acc.append(val_epoch_acc)
    # store epoch results in prettytable row
    table.add_rows([ round(_, 4) for _ in [epoch+1, train_epoch_loss, val_epoch_loss, train_epoch_acc, val_epoch_acc]])
print(table)

# --- RESULTS VISUALIZATION ---
fig, ax = plt.subplots(1, 1, figsize=(16, 5))  

ax.plot(mobile_val_loss, label='MobileNetV3(Small) Loss', color='red')


ax.set_title('Validation Loss vs Epochs')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True)
