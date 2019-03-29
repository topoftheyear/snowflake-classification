import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tqdm import tqdm
import numpy as np
from spp_layer import *

def get_model(image_size):
    
    model = tf.keras.Sequential()

    # Layer 1
    model.add(keras.layers.Conv2D(
        2,
        (2, 2),
        padding="same",
        input_shape=image_size,
        activation="relu",
    ))
    model.add(keras.layers.MaxPooling2D(
        pool_size=(2, 2),
    ))

    # Layer 2
    model.add(keras.layers.Conv2D(
        4,
        (2, 2),
        padding="same",
        activation="relu",
    ))
    model.add(keras.layers.MaxPooling2D(
        pool_size=(2, 2),
    ))
    
    # Layer 3
    model.add(keras.layers.Conv2D(
        8,
        (2, 2),
        padding="same",
        activation="relu",
    ))
    
    # Pyramid layer
    model.add(keras.layers.Lambda(spatial_pyramid_pool, pyramid_output))

    # Layer 5
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(
        200,
        activation="relu",
    ))

    # Layer 6
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
    
    return model
