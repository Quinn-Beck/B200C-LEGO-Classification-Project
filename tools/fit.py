import torch

def train(model, dataloader, optimizer, loss_fn):
    model.train()
    counter = 0
    train_running_loss = 0.0
    correct = 0
    total = len(dataloader.dataset)
    for i, (images, labels) in enumerate(dataloader):
        counter += 1
        # load data to gpu
        images = images.to(device)
        labels = labels.to(device)
        
        y = model(images)
        loss = loss_fn(y, labels)
        train_running_loss += loss.item()
        
        values, preds = torch.max(y, 1)
        # count correctly classified images
        correct += torch.sum(preds == labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # training loss and accuracy for the epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100.0 * (correct / total)
    
    return epoch_loss, epoch_acc

def test(model, dataloader, loss_fn):
    model.eval()
    counter = 0
    correct = 0
    total = len(dataloader.dataset)

    test_running_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            counter += 1
            # load data to gpu
            images = images.to(device)
            labels = labels.to(device)
            
            y = model(images)
            loss = loss_fn(y, labels)
            val_running_loss += loss.item()
            
            values, preds = torch.max(y, 1)
            # count correctly classified images
            correct += torch.sum(preds == labels)

            # Collect all targets and predictions for metric calculations
            all_labels.extend(labels.view_as(predicted).cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate overall metrics
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    # test loss and accuracy for the epoch
    test_loss = test_running_loss / counter
    accuracy = 100.0 * (correct / total)
    
    return test_loss, accuracy, precision, recall, f1

def validate(model, dataloader, loss_fn):
    model.eval()
    counter = 0
    val_running_loss = 0.0
    correct = 0
    total = len(dataloader.dataset)
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            counter += 1
            # load data to gpu
            images = images.to(device)
            labels = labels.to(device)
            
            y = model(images)
            loss = loss_fn(y, labels)
            val_running_loss += loss.item()
            
            values, preds = torch.max(y, 1)
            # count correctly classified images
            correct += torch.sum(preds == labels)
    # validation loss and accuracy for the epoch
    epoch_loss = val_running_loss / counter
    epoch_acc = 100.0 * (correct / total)
    
    return epoch_loss, epoch_acc
