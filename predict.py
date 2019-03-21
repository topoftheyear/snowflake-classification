from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tqdm import tqdm
import numpy as np
from helpers import *

tf.enable_eager_execution()


def main():
    # read in data and labels from files
    print('reading input from files')
    test_im_set = []
    test_label_set = []

    with open("test_imageset", "r") as f:
        for line in f:
            test_im_set.append(line.replace('\n', ''))

    with open("test_labelset", "r") as f:
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

    test_image_labels = [label_to_index[label] for label in test_label_set]
    test_image_labels = keras.utils.to_categorical(test_image_labels, 4)

    test_image_paths = []
    for image in tqdm(test_im_set):
        test_image_paths.append(load_and_preprocess_image(path=image))

    model = keras.models.load_model('model.h5')

    # Compiling the model
    model.compile(
        optimizer=OPTIMIZER,
        loss=LOSS,
        metrics=METRICS
    )

    results = []
    for image in tqdm(test_image_paths):
        res = model.predict(
            x=image,
            verbose=0,
        )
        results.append(res)

    print(results)

    with open('results.txt', 'w+') as f:
        for (x,y) in np.ndenumerate(results):
            string = "Image " + str(x[0]) + ": " + index_to_label[x[2]] + ": " + "{0:.0%}".format(y) + "\n"
            f.write(string)
            
    with open('results_unedited.txt', 'w+') as f:
        for (x,y) in np.ndenumerate(results):
            f.write(str(x) + " | " + str(y) + "\n")


if __name__ == "__main__":
    main()
