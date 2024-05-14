import PIL
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder

def data_generator(train_t, eval_t, num_classes = 10):
    """
    Prepare images for batch streaming.
    """
    full_train_data = ImageFolder(root = "C:/Users/Quinn/Desktop/B200C-Lego-Classification/train",
                            transform = train_t, loader = PIL.Image)
    full_val_data = ImageFolder(root = "C:/Users/Quinn/Desktop/B200C-Lego-Classification/val",
                            transform = eval_t, loader = PIL.Image)
    full_test_data = ImageFolder(root = "C:/Users/Quinn/Desktop/B200C-Lego-Classification/test",
                            transform = eval_t, loader = PIL.Image)
    
    train_data = Subset(full_train_data, [0:3200*num_classes])
    val_data = Subset(full_val_data, [0:400*num_classes])
    test_data = Subset(full_test_data, [0:400*num_classes])
    
    return train_data, val_data, test_data
