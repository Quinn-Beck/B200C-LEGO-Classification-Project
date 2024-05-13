import os
import glob
import PIL

# rename `64` folder containing all images to the training data folder
os.rename("C:/Users/Quinn/Desktop/B200C-Lego-Classification/64", "C:/Users/Quinn/Desktop/B200C-Lego-Classification/train")

# create `val` folder
val_directory = "C:/Users/Quinn/Desktop/B200C-Lego-Classification/val"
os.makedirs(val_directory, exist_ok=True)

# create `test` folder
test_directory = "C:/Users/Quinn/Desktop/B200C-Lego-Classification/test"
os.makedirs(test_directory, exist_ok=True)

for folder in glob.glob("C:/Users/Quinn/Desktop/B200C-Lego-Classification/train/*"):
    images = glob.glob(f"{folder}/*.jpg")
    label = folder.split("/")[-1]

    val_images = [PIL.Image(img) for img in images[3200:3600]]
    val_subdir = val_directory.join(f"/{label}")
    for i, img in enumerate(val_images):
        img.save(os.path.join(val_subdir, f'image_{i}.jpg'))
  
    test_images = [PIL.Image(img) for img in images[3600:]]
    test_subdir = test_directory.join(f"/{label}")
    for i, img in enumerate(test_images):
        img.save(os.path.join(test_subdir, f'image_{i}.jpg'))
