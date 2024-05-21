import PIL
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder

def data_generator(train_t, eval_t, num_classes = 10):
    """
    Prepare images for batch streaming.
    """
    train_data = ImageFolder(root = "C:/Users/Quinn/Desktop/B200C-Lego-Classification/train",
                            transform = train_t)
    val_data = ImageFolder(root = "C:/Users/Quinn/Desktop/B200C-Lego-Classification/val",
                            transform = eval_t)
    test_data = ImageFolder(root = "C:/Users/Quinn/Desktop/B200C-Lego-Classification/test",
                            transform = eval_t)
    
    if num_classes != 200:
        train_data = Subset(train_data, range(3200*num_classes))
        val_data = Subset(val_data, range(400*num_classes))
        test_data = Subset(test_data, range(400*num_classes))
    
    return train_data, val_data, test_data