from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
from spp_layer import *
from tqdm import tqdm
import numpy as np
from helpers import *
import math
from random import shuffle
#import winsound

EPOCHS = 5
BATCH_SIZE = 1
STEPS_PER_EPOCH = 500
SECTIONS = 36

image_count = 0

tf.enable_eager_execution()

print(tf.__version__)

def main():
    # read in data and labels from files
    print('reading input from files')
    im_set = []
    la_set = []
    with open("imageset", "r") as f:
        for line in f:
            im_set.append(line.replace('\n', ''))
    image_count = len(im_set)
    STEPS_PER_EPOCH = image_count

    with open("labelset", "r") as f:
        for line in f:
            la_set.append(line.replace('\n', ''))

    label_to_index = {
        "Bullets_BulletRosettes": 0,
        "Columns": 1,
        "Dendrite": 2,
        "Plates": 3,
    }

    #all_image_paths = []
    #for image in tqdm(im_set):
    #    all_image_paths.append(load_and_preprocess_image(path=image))

    #path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

    #label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.float32))

    #image_label_ds = tf.data.Dataset.zip((path_ds, label_ds))

    #print('image shape: ', image_label_ds.output_shapes[0])
    #print('label shape: ', image_label_ds.output_shapes[1])
    #print('types: ', image_label_ds.output_types)
    #print()
    #print(image_label_ds)

    #ds = image_label_ds.shuffle(buffer_size=image_count)
    #ds = ds.repeat()
    # ds = ds.batch(BATCH_SIZE)
    # ds = ds.prefetch(buffer_size=2)
    # print(ds)

    model = tf.keras.Sequential()

    # Layer 1
    model.add(keras.layers.Conv2D(
        2,
        (2, 2),
        padding="same",
        input_shape=(None, None, 3),
        activation="relu",
    ))
    model.add(keras.layers.MaxPooling2D(
        pool_size=(4, 4),
        strides=(2, 2),
    ))

    # Layer 2
    #model.add(keras.layers.Conv2D(
    #    4,
    #    (4, 4),
    #    padding="same",
    #    activation="relu",
    #))
    #model.add(keras.layers.MaxPooling2D(
    #    pool_size=(4, 4),
    #    strides=(4, 4),
    #))
    
    # Layer 3
    #model.add(keras.layers.Conv2D(
    #    8,
    #    (8, 8),
    #    padding="same",
    #    activation="relu",
    #))
    #model.add(keras.layers.MaxPooling2D(
    #    pool_size=(8, 8),
    #    strides=(8, 8),
    #))
    
    # Layer 4
    #model.add(keras.layers.Conv2D(
    #    2,
    #    (2, 2),
    #    padding="same",
    #    activation="relu",
    #))
    #model.add(keras.layers.MaxPooling2D(
    #    pool_size=(4, 4),
    #    strides=(4, 4),
    #))
    
    # Pyramid layer
    model.add(keras.layers.Lambda(spatial_pyramid_pool, pyramid_output))
    
    #model.add(keras.layers.MaxPooling2D(
    #    pool_size=(1,1),
    #    strides=(1,1),
    #))

    # Layer 6
    #model.add(keras.layers.Reshape((17280,)))
    model.add(keras.layers.Flatten())
    #model.add(keras.layers.Dense(
    #    60,
    #    activation="relu",
    #))

    # Layer 7
    model.add(keras.layers.Dense(
        4,
        activation="softmax",
    ))

    # Compiling the model
    model.compile(
        optimizer=OPTIMIZER,
        loss=LOSS,
        metrics=METRICS,
    )

    print(model.summary())
    #input()

    # Things to do per epoch
    for epoch_num in tqdm(range(EPOCHS)):
        #print('Epoch ' + str(epoch_num + 1) + '/' + str(EPOCHS))
        
        # Shuffle data
        '''im_set_shuf = []
        la_set_shuf = []
        index_shuf = list(range(len(im_set)))
        shuffle(index_shuf)
        #print("Shuffling")
        for i in index_shuf:
            im_set_shuf.append(im_set[i])
            la_set_shuf.append(la_set[i])
            
        im_set = im_set_shuf
        la_set = la_set_shuf'''

        all_image_labels = [label_to_index[label] for label in la_set]
        all_image_labels = keras.utils.to_categorical(all_image_labels, 4)
        
        # Things to do per section
        #for section in range(0, SECTIONS):
        for x in range(len(im_set)):
            image = im_set[x]
            label = label_to_index[la_set[x]]
            
            all_image_paths = []
            all_label_paths = []
            path_ds = None
            label_ds = None
            # Gather all images within the section given a chunk
            #print("Reading in files")
            #for x in tqdm(range(math.floor((section - 1) * len(all_image_labels) / SECTIONS), math.floor(section * len(all_image_labels) / SECTIONS))):
            all_image_paths.append(load_and_preprocess_image(path=image))
            all_label_paths.append(label)
        
            path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

            label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_label_paths, tf.float32))

            image_label_ds = tf.data.Dataset.zip((path_ds, label_ds))

            #print('image shape: ', image_label_ds.output_shapes[0])
            #print('label shape: ', image_label_ds.output_shapes[1])
            #print('types: ', image_label_ds.output_types)
            #print()
            #print(image_label_ds)

            #ds = image_label_ds.shuffle(buffer_size=image_count)
            ds = image_label_ds.repeat()
            
            #print(ds.output_shapes)
            
            model.fit(
                x=ds,
                batch_size=BATCH_SIZE,
                epochs=1,
                verbose=0,
                #validation_split=0.05,
                shuffle=False,
                steps_per_epoch=1,
            )
            
            all_image_paths = []
            all_label_paths = []
            
        temp_test(epoch_num, model)

    model.save('model.h5')


if __name__ == "__main__":
    try:
        main()
    except ValueError:
        #winsound.PlaySound(f'Debug/oh no.wav', winsound.SND_ALIAS)
        print(*ValueError.args)
    except TypeError:
        #winsound.PlaySound(f'Debug/oof.wav', winsound.SND_ALIAS)
        print(*TypeError.args)
    except Exception:
        #winsound.PlaySound(f'Debug/toilet.wav', winsound.SND_ALIAS)
        print(*Exception.args)
