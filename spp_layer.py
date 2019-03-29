import numpy as np
import math
import tensorflow as tf
from helpers import *


# input feature maps is of the form: N-C-(WH)/(HW)
# ex. spatial_pyramid:
#	[[1, 1], [2, 2], [3, 3], [4, 5]]
# each row is a level of pyramid with nxm pooling
def np_spatial_pyramid_pooling(input_feature_maps, spatial_pyramid, dtype=np.float32):
    assert input_feature_maps.ndim == 4
    assert spatial_pyramid.ndim == 2
    assert spatial_pyramid.shape[1] == 2

    batch_size = input_feature_maps.shape[0]
    num_channels = input_feature_maps.shape[1]
    h = input_feature_maps.shape[2]
    w = input_feature_maps.shape[3]

    h = int(h)
    w = int(w)

    num_levels = spatial_pyramid.shape[0]

    # N-C-W*H
    flattened_feature_maps = np.reshape(input_feature_maps, (batch_size, num_channels, -1))
    num_px = flattened_feature_maps.shape[2]

    bins_per_level = np.prod(spatial_pyramid, axis=1)
    num_bins = np.sum(bins_per_level)
    stack = []

    # stride tricks, then max pool along one dimension
    # then stride tricks again and max pool along the other dimension
    # but whats the length and stride?
    # ceil(w/n) for window size, floor(w/n) for stride,
    # where w is the original dim, and n is the number of bins along the dim
    # but this implementation may leave out some pixels (consider w = 5, n = 3)

    sizeof_item = np.dtype(dtype).itemsize

    for i in range(num_levels):
        n_h = spatial_pyramid[i][0]
        n_w = spatial_pyramid[i][1]

        l = math.ceil(w / n_w)
        s = math.floor(w / n_w)

        ar = np.lib.stride_tricks.as_strided(flattened_feature_maps, (batch_size, num_channels, h, n_w, l),
                                             (
                                             sizeof_item * num_px * num_channels, sizeof_item * num_px, sizeof_item * w,
                                             sizeof_item * s, sizeof_item))
        ar = np.transpose(np.amax(ar, axis=4), (0, 1, 3, 2)).copy()

        l = math.ceil(h / n_h)
        s = math.floor(h / n_h)

        ar = np.lib.stride_tricks.as_strided(ar, (batch_size, num_channels, n_w, n_h, l),
                                             (sizeof_item * n_w * h * num_channels, sizeof_item * n_w * h,
                                              sizeof_item * h, sizeof_item * s, sizeof_item))
        ar = np.transpose(np.amax(ar, axis=4), (0, 1, 3, 2))

        # for debugging purposes, monitor "ar" here, before flattening

        stack.append(np.reshape(ar, (batch_size, num_channels, -1)))

    stack = np.concatenate(stack, axis=2)
    #print(stack.shape)

    return stack


def tf_spatial_pyramid_pooling(tf_input_feature_maps, tf_spatial_pyramid, dtype=tf.float32):
    return tf.py_func(np_spatial_pyramid_pooling, [tf_input_feature_maps, tf_spatial_pyramid], dtype)
 
 
def max_pool_2d_nxn_regions(inputs, output_size: int, mode: str):
    """
    Performs a pooling operation that results in a fixed size:
    output_size x output_size.
    
    Used by spatial_pyramid_pool. Refer to appendix A in [1].
    
    Args:
        inputs: A 4D Tensor (B, H, W, C)
        output_size: The output size of the pooling operation.
        mode: The pooling mode {max, avg}
        
    Returns:
        A list of tensors, for each output bin.
        The list contains output_size * output_size elements, where
        each elment is a Tensor (N, C).
        
    References:
        [1] He, Kaiming et al (2015):
            Spatial Pyramid Pooling in Deep Convolutional Networks
            for Visual Recognition.
            https://arxiv.org/pdf/1406.4729.pdf.
            
    Ported from: https://github.com/luizgh/Lasagne/commit/c01e3d922a5712ca4c54617a15a794c23746ac8c
    """
    inputs_shape = tf.shape(inputs)
    h = tf.cast(tf.gather(inputs_shape, 1), tf.int32)
    w = tf.cast(tf.gather(inputs_shape, 2), tf.int32)
    
    if mode == 'max':
        pooling_op = tf.reduce_max
    elif mode == 'avg':
        pooling_op = tf.reduce_mean
    else:
        msg = "Mode must be either 'max' or 'avg'. Got '{0}'"
        raise ValueError(msg.format(mode))
        
    result = []
    n = output_size
    for row in range(output_size):
        for col in range(output_size):
            # start_h = floor(row / n * h)
            start_h = tf.cast(tf.floor(tf.multiply(tf.divide(row, n), tf.cast(h, tf.float32))), tf.int32)
            # end_h = ceil((row + 1) / n * h)
            end_h = tf.cast(tf.ceil(tf.multiply(tf.divide((row + 1), n), tf.cast(h, tf.float32))), tf.int32)
            # start_w = floor(col / n * w)
            start_w = tf.cast(tf.floor(tf.multiply(tf.divide(col, n), tf.cast(w, tf.float32))), tf.int32)
            # end_w = ceil((col + 1) / n * w)
            end_w = tf.cast(tf.ceil(tf.multiply(tf.divide((col + 1), n), tf.cast(w, tf.float32))), tf.int32)
            pooling_region = inputs[:, start_h:end_h, start_w:end_w, :]
            pool_result = pooling_op(pooling_region, axis=(1, 2))
            result.append(pool_result)
    return result


def spatial_pyramid_pool(inputs, dimensions=[1, 2, 3, 4], mode='max', implementation='kaiming'):
    """
    Performs spatial pyramid pooling (SPP) over the input.
    It will turn a 2D input of arbitrary size into an output of fixed
    dimenson.
    Hence, the convlutional part of a DNN can be connected to a dense part
    with a fixed number of nodes even if the dimensions of the input
    image are unknown.
    
    The pooling is performed over :math:`l` pooling levels.
    Each pooling level :math:`i` will create :math:`M_i` output features.
    :math:`M_i` is given by :math:`n_i * n_i`, with :math:`n_i` as the number
    of pooling operations per dimension level :math:`i`.
    
    The length of the parameter dimensions is the level of the spatial pyramid.
    
    Args:
        inputs: A 4D Tensor (B, H, W, C).
        dimensions: The list of :math:`n_i`'s that define the output dimension
        of each pooling level :math:`i`. The length of dimensions is the level of
        the spatial pyramid.
        mode: Pooling mode 'max' or 'avg'.
        implementation: The implementation to use, either 'kaiming' or 'fast'.
        kamming is the original implementation from the paper, and supports variable
        sizes of input vectors, which fast does not support.
    
    Returns:
        A fixed length vector representing the inputs.
    
    Notes:
        SPP should be inserted between the convolutional part of a DNN and it's
        dense part. Convolutions can be used for arbitrary input dimensions, but
        the size of their output will depend on their input dimensions.
        Connecting the output of the convolutional to the dense part then
        usually demands us to fix the dimensons of the network's input.
        The spatial pyramid pooling layer, however, allows us to leave 
        the network input dimensions arbitrary. 
        The advantage over a global pooling layer is the added robustness 
        against object deformations due to the pooling on different scales.
        
    References:
        [1] He, Kaiming et al (2015):
            Spatial Pyramid Pooling in Deep Convolutional Networks
            for Visual Recognition.
            https://arxiv.org/pdf/1406.4729.pdf.
            
    Ported from: https://github.com/luizgh/Lasagne/commit/c01e3d922a5712ca4c54617a15a794c23746ac8c
    """
    pool_list = []
    if implementation == 'kaiming':
        for pool_dim in dimensions:
            pool_list += max_pool_2d_nxn_regions(inputs, pool_dim, mode)
    else:
        shape = inputs.get_shape().as_list()
        for d in dimensions:
            h = shape[1]
            w = shape[2]
            ph = np.ceil(h * 1.0 / d).astype(np.int32)
            pw = np.ceil(w * 1.0 / d).astype(np.int32)
            sh = np.floor(h * 1.0 / d + 1).astype(np.int32)
            sw = np.floor(w * 1.0 / d + 1).astype(np.int32)
            pool_result = tf.nn.max_pool(inputs,
                                         ksize=[1, ph, pw, 1], 
                                         strides=[1, sh, sw, 1],
                                         padding='SAME')
            pool_list.append(tf.reshape(pool_result, [tf.shape(inputs)[0], -1]))
    return tf.concat(pool_list, 1)


def pyramid_output(input_shape):
    shape = list(input_shape)
    shape[-1] *= 14
    return shape
    