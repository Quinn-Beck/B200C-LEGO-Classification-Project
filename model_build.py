import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from torchvision.models import mobilenet_v3_small
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from preprocessing_pipeline_v2 import data_generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: add more transforms - RandomPerspective/ RandomRotation/ FiveCrop OR RandomCrop &OR RandomResizedCrop
transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

# start small scale with only 10 classes
num_classes = 10

# --- SOME HYPERPARAMETERS ---
num_workers = [0,1,2,3]
num_epochs = 5
learn_rate = 1e-4

# --- LOAD DATA ---
train_data, val_data, test_data = data_generator(num_classes, transform)



# --- INSTANTIATE / LOAD MODEL ---
# model.load()???
mobilenet_model = mobilenet_v3_small(weights='DEFAULT')
# get number of in features from source
num_features = mobilenet_model.classifier[3].in_features
# redefine the networks final fully connected layer
mobilenet_model.classifier[3] = nn.Linear(num_features, num_classes)
# send to gpu
mobilenet_model = mobilenet_model.to(device)

# --- TRAINING ---
# TODO: add in validation data, early stopping, etc.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=mobilenet_model.parameters(), lr = learn_rate)

# initialize training loss
training_loss = []

for epoch in range(num_epochs):
    start_time = time.time()
    for i, (images, labels) in enumerate(train_dl):
        # set model to training mode
        mobilenet_model.train()
        # load data to gpu
        images = images.to(device)
        labels = labels.to(device)
        
        y_pred = mobilenet_model(images)
        loss = mobile_loss_fn(y_pred, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # set model to eval mode
        # mobilenet_model.eval()
      
        if i % 5000 == 0:
            training_loss.append(loss.item())
            print(f"Epoch: {epoch+1}/{num_epochs} --- Training Loss: {loss.item():.4f}")
            
mobile_train_time = time.time() - start_time
# model.save()???
