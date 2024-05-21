import torch
import sklearn.metrics as metrics
import numpy as np

# custom accuracy computation, optionally displays predictions
def validate(model, data, display_pred = False):
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(data):
        images = images.to(device)
        labels = labels.to(device)
        
        model.eval()
        x = model(images)
        value, pred = torch.max(x, 1)
        
        total += x.size(0)
        correct += torch.sum(pred == labels)
        
        if i % 1000 == 0 & display_pred == True:
            print(f"Pred: {x} / True: {labels}")
    return correct / total

def confusion_plot(model, data):
    y_true = []
    y_pred = []
    for i, (images, labels) in enumerate(data):

        model.eval()
        x = model(images)
        value, pred = torch.max(x, 1)
        
        y_pred += pred
        y_true += labels

    confusion_df = pd.DataFrame(confusion_matrix(y_true, y_pred), 
                                index = [j for j in range(10)],
                                columns = [j for j in range(10)]
                               )
  
    plt.figure(figsize = (10,7))
    sns.heatmap(confusion_df, annot=True)
    plt.show()

# return test accuracy, precision recall, f1 score, (# of parameters? inference time?) for a trained model
def scores(model, data):
    