from sklearn.metrics import top_k_accuracy_score as top_k

# --- CHECKPOINT TOOLS ---
def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    
def resume(model, filename):
    model.load_state_dict(torch.load(filename))

# --- LOAD DATA ---
train_data, val_data, test_data = data_generator(train_transform, 
                                                 eval_transform, 
                                                 num_classes)

train_dl = torch.utils.data.DataLoader(train_data, batch_size=32,
                                            shuffle=True, num_workers=2)
val_dl = torch.utils.data.DataLoader(val_data, batch_size=32,
                                            shuffle=True, num_workers=2)
test_dl = torch.utils.data.DataLoader(test_data, batch_size=32,
                                            shuffle=True, num_workers=2)

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
optimizer = torch.optim.Adam(params=mobilenet_model.parameters(), lr=learn_rate)

# initialize training loss
training_loss = []

for epoch in range(num_epochs):
    #start_time = time.time()
    i = 0
    for (images, labels) in train_dl:
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

        # counter ticks up
        i += 1
            
#mobile_train_time = time.time() - start_time
# model.save()???
