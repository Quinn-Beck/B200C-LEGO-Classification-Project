resnet_model = resnet50(weights='DEFAULT', num_classes = num_classes)
resnet_model.fc = nn.Linear(512 * block.expansion, num_classes)
resnet_model = squeezenet_model.to(device)

# print(next(resnet_model.parameters()).device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=resnet_model.parameters(), lr = 1e-5)

training_loss = []
num_epochs = 20
num_training_steps = num_epochs * len(train_dataloader)

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    for i, (b_x, b_y) in enumerate(train_dataloader):
        # load data to gpu
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        
        y_pred = resnet_model(b_x)
        loss = loss_fn(y_pred, b_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # progress_bar.update(1)
        
        if i % 100 == 0:
            training_loss.append(loss.item())
            print(f"Batch: {i} / Epoch: {epoch} / Loss: {loss.item():.4f} / Pred:{y_pred[0]}")