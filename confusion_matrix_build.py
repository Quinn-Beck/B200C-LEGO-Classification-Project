import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def confusion_plot(model, data):
    y_true = []
    y_pred = []
    for i, (images, labels) in enumerate(data):
        images = images.to(device)
        labels = labels.to(device)

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
