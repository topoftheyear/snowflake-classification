import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tqdm import tqdm
import numpy as np
    
EPOCHS = 50
BATCH_SIZE = 1
STEPS_PER_EPOCH = 500  # doesn't matter actually
SECTIONS = 40

USE_PREVIOUS = True
    
OPTIMIZER = keras.optimizers.Adam(lr=0.001)
LOSS = keras.losses.categorical_crossentropy
METRICS = [keras.metrics.categorical_crossentropy]

PYRAMID_SHAPE_OTHER = [1, 2, 3, 4]
PYRAMID_RETURN_SIZE = 30


def preprocess_image(image, channels):
    image = tf.image.decode_png(image, channels, dtype=tf.uint8)
    #image = tf.expand_dims(image, axis=3)
    #image = np_spatial_pyramid_pooling(
    #    input_feature_maps=image,
    #    spatial_pyramid=np.array(PYRAMID_SHAPE),
    #    dtype=np.uint8,
    #)
    image = K.cast(image, dtype='float32')
    image = tf.expand_dims(image, axis=0)
    #print(image.shape)

    return image


def load_and_preprocess_image(path, channels=0):
    image = tf.read_file(path)
    return preprocess_image(image, channels)
    
    
