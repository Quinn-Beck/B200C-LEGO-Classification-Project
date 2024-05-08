import PIL
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder

def data_generator(num_classes = 10, t):
    """
    Prepare images for batch streaming.
    """
    full_data = ImageFolder(root = "C:/Users/Quinn/Desktop/B200C-Lego-Classification/64/*",
                            transform = t, loader = PIL.Image)
    
    # get indices to slice the full dataset on to separate training, validation, and testing data
    # TODO: test that images are read in by ImageFolder sequentially
    train_ids = [idx for n in range(num_classes) for idx in range(4000*n, 4000*n+3200)]
    val_ids = [idx for n in range(num_classes) for idx in range(4000*n+3200, 4000*n+3600)]
    test_ids = [idx for n in range(num_classes) for idx in range(4000*n+3600, 4000*(n+1))]

    train_data = Subset(full_data, train_ids)
    val_data = Subset(full_data, val_ids)
    test_data = Subset(full_data, test_ids)

    return train_data, val_data, test_data
