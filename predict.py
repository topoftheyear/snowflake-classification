import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tqdm import tqdm
import numpy as np
from helpers import *
from model import *
from spp_layer import *

#tf.enable_eager_execution()


def main():
    # read in data and labels from files
    print('reading input from files')
    test_im_set = []
    test_label_set = []

    with open("imageset", "r") as f:
        for line in f:
            test_im_set.append(line.replace('\n', ''))

    with open("labelset", "r") as f:
        for line in f:
            test_label_set.append(line.replace('\n', ''))

    label_to_index = {
        "Bullets_BulletRosettes": 0,
        "Columns": 1,
        "Dendrite": 2,
        "Plates": 3,
    }
    
    index_to_label = {
        0: "Bullets_BulletRosettes",
        1: "Columns",
        2: "Dendrite",
        3: "Plates",
    }

    temp_image_labels = [label_to_index[label] for label in test_label_set]
    test_image_labels = keras.utils.to_categorical(temp_image_labels, 4)

    test_image_paths = []
    for image in tqdm(test_im_set):
        test_image_paths.append(load_and_preprocess_image(path=image, channels=1))

    model = keras.models.load_model('model.h5', custom_objects={'Pyramid': spatial_pyramid_pool})

    results = []
    for image in tqdm(test_image_paths):
        res = model.predict(
            x=image,
            verbose=0,
            steps=1,
        )
        results.append(res)
    
    formatted = {}
    guessed = []
    correct = 0
    incorrect = 0

    with open('results.txt', 'w+') as f:
        for (x,y) in np.ndenumerate(results):
            string = "Image " + str(x[0]) + ": " + index_to_label[x[2]] + ": " + "{0:.0%}".format(y) + "\n"
            f.write(string)
            
            if x[0] not in formatted:
                formatted[x[0]] = {}
            formatted[x[0]][x[2]] = y
            
    with open('results_unedited.txt', 'w+') as f:
        for (x,y) in np.ndenumerate(results):
            f.write(str(x) + " | " + str(y) + "\n")
            
    # Get best guess for each image
    for x in range(len(test_image_paths)):
        group = formatted[x]
        maximum = max(group, key=group.get)
        guessed.append({maximum: group[maximum]})
        
    # Compare best guess to actual result
    for x in range(len(guessed)):
        if temp_image_labels[x] in guessed[x].keys():
            correct += 1
        else:
            incorrect += 1
    
    print("Correct: " + str(correct) + " | " + "Incorrect: " + str(incorrect))
    
    accuracy = correct / (incorrect + correct)
    print("Accuracy: " + str(accuracy))


if __name__ == "__main__":
    main()
