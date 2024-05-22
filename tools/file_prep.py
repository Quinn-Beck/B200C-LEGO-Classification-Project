import os
import shutil
from os import path

'''
This script was used to split the data into train, val, and test
sets directly within the filesystem. This was a necessary step
in order to use the torchvision.ImageFolder class later on.
'''

# rename original `64` folder containing all images to `train`
train_directory = "C:/Users/Quinn/Desktop/B200C-Lego-Classification/train"
# os.rename("C:/Users/Quinn/Desktop/B200C-Lego-Classification/64", train_directory)

# create `val` folder
val_directory = "C:/Users/Quinn/Desktop/B200C-Lego-Classification/val"
os.makedirs(val_directory, exist_ok=True)

# create `test` folder
test_directory = "C:/Users/Quinn/Desktop/B200C-Lego-Classification/test"
os.makedirs(test_directory, exist_ok=True)

# for each class folder
for label in os.listdir(train_directory):
    train_subdir = path.join(train_directory, label).replace("\\","/")
    # get images
    images = os.listdir(train_subdir)

    # create class subdirectory in the validation folder
    val_subdir = path.join(val_directory, label)
    os.makedirs(val_subdir, exist_ok=True)
    for img in images[3200:3600]:
        # copy images from the respective training subdirectory
        # into the new validation subdirectory
        source = path.join(train_subdir, img).replace("\\","/")
        destination = path.join(val_subdir, img).replace("\\","/")
        shutil.copy(source, destination)
        # delete originals from training data
        os.remove(source)
    
    # create class subdirectory in the test folder
    test_subdir = path.join(test_directory, label)
    os.makedirs(test_subdir, exist_ok=True)
    for img in images[3600:]:
        # copy images from the respective training subdirectory
        # into the new test subdirectory
        source = path.join(train_subdir, img).replace("\\","/")
        destination = path.join(test_subdir, img).replace("\\","/")
        shutil.copy(source, destination)
        # delete originals from training data
        os.remove(source)