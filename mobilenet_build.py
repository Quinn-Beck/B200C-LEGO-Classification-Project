mobilenet_model = mobilenet_v3_large(weights='DEFAULT')
# get number of in features from source
num_features = mobilenet_model.classifier[3].in_features
# redefine the networks final fully connected layer
mobilenet_model.classifier[3] = nn.Linear(num_features, num_classes)
# send to gpu
mobilenet_model = mobilenet_model.to(device)


mobile_loss_fn = nn.CrossEntropyLoss()
mobile_optimizer = torch.optim.Adam(params=mobilenet_model.parameters(), lr = 1e-5)

# initialize training loss
mobile_training_loss = []


for epoch in range(num_epochs):
    for i, (b_x, b_y) in enumerate(train_dataloader):
        # load data to gpu
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        
        y_pred = mobilenet_model(b_x)
        loss = mobile_loss_fn(y_pred, b_y)
        
        mobile_optimizer.zero_grad()
        loss.backward()
        mobile_optimizer.step()
        
        if i % 5000 == 0:
            mobile_training_loss.append(loss.item())
            print(f"Epoch: {epoch+1}/{num_epochs} --- Training Loss: {loss.item():.4f} --- Training Accuracy: ")
