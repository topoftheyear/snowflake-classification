from os import listdir
from os.path import *
from tqdm import tqdm

data_location = "Corrected_Images"

images = []
labels = []

test_images = []
test_labels = []

for f in tqdm(listdir(data_location)):
    test = True
    for file in listdir(f"{data_location}/{f}"):
        image = file
        if False:
            test_images.append(f"{data_location}/{f}/{image}")
            test_labels.append(f)
            test = False
        else:
            images.append(f"{data_location}/{f}/{image}")
            labels.append(f)

with open("imageset", "w+") as f:
    for image in tqdm(images):
        f.write(image + '\n')

with open("labelset", "w+") as f:
    for label in tqdm(labels):
        f.write(label + '\n')

with open("test_imageset", "w+") as f:
    for image in tqdm(test_images):
        f.write(image + '\n')

with open("test_labelset", "w+") as f:
    for label in tqdm(test_labels):
        f.write(label + '\n')
