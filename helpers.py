import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tqdm import tqdm
import numpy as np
    
OPTIMIZER = tf.train.AdamOptimizer(learning_rate=0.001)
LOSS = keras.losses.categorical_crossentropy
METRICS = [keras.metrics.categorical_crossentropy]

PYRAMID_SHAPE = [[1, 1], [2, 2], [3, 3], [4, 4]]
PYRAMID_SHAPE_OTHER = [1, 2, 3, 4]


def preprocess_image(image):
    image = tf.image.decode_png(image, channels=3, dtype=tf.uint8)
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


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)
    