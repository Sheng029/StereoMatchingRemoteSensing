import tensorflow.keras as keras

L2 = 1.0e-4


def conv2d(filters, kernel_size, strides, padding, dilation_rate, use_bias=False):
    """
    2D convolution.
    param use_bias: whether to use bias after convolution operation.
    """
    return keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                               padding=padding, dilation_rate=dilation_rate, activation=None,
                               use_bias=use_bias, kernel_initializer='he_normal',
                               kernel_regularizer=keras.regularizers.l2(L2))


def conv2d_bn_act(filters, kernel_size, strides, padding, dilation_rate, activation=None):
    """
    2D convolution followed by bn and activation.
    param activation: 'relu', 'leaky_relu' or None.
    """
    conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                               padding=padding, dilation_rate=dilation_rate, activation=None,
                               use_bias=False, kernel_initializer='he_normal',
                               kernel_regularizer=keras.regularizers.l2(L2))
    bn = keras.layers.BatchNormalization()
    relu = keras.layers.ReLU()
    leaky_relu = keras.layers.LeakyReLU()

    if activation == 'relu':
        return keras.Sequential([conv, bn, relu])
    elif activation == 'leaky_relu':
        return keras.Sequential([conv, bn, leaky_relu])
    else:
        return keras.Sequential([conv, bn])


def conv3d(filters, kernel_size, strides, padding, use_bias=False):
    """
    3D convolution.
    param use_bias: whether to use bias after convolution operation.
    """
    return keras.layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=strides,
                               padding=padding, activation=None, use_bias=use_bias,
                               kernel_initializer='he_normal',
                               kernel_regularizer=keras.regularizers.l2(L2))


def conv3d_bn_act(filters, kernel_size, strides, padding, activation=None):
    """
    3D convolution followed by bn and activation.
    param activation: 'relu', 'leaky_relu', or None.
    """
    conv = keras.layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=strides,
                               padding=padding, activation=None, use_bias=False,
                               kernel_initializer='he_normal',
                               kernel_regularizer=keras.regularizers.l2(L2))
    bn = keras.layers.BatchNormalization()
    relu = keras.layers.ReLU()
    leaky_relu = keras.layers.LeakyReLU()

    if activation == 'relu':
        return keras.Sequential([conv, bn, relu])
    elif activation == 'leaky_relu':
        return keras.Sequential([conv, bn, leaky_relu])
    else:
        return keras.Sequential([conv, bn])


def conv3d_transpose(filters, kernel_size, strides, padding, use_bias=False):
    """
    Transpose 3D convolution.
    param use_bias: whether to use bias after convolution operation.
    """
    return keras.layers.Conv3DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                                        padding=padding, activation=None, use_bias=use_bias,
                                        kernel_initializer='he_normal',
                                        kernel_regularizer=keras.regularizers.l2(L2))


def conv3d_transpose_bn_act(filters, kernel_size, strides, padding, activation=None):
    """
    Transpose 3D convolution followed by bn and activation.
    param activation: 'relu', 'leaky_relu', or None.
    """
    conv = keras.layers.Conv3DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                                        padding=padding, activation=None, use_bias=False,
                                        kernel_initializer='he_normal',
                                        kernel_regularizer=keras.regularizers.l2(L2))
    bn = keras.layers.BatchNormalization()
    relu = keras.layers.ReLU()
    leaky_relu = keras.layers.LeakyReLU()

    if activation == 'relu':
        return keras.Sequential([conv, bn, relu])
    elif activation == 'leaky_relu':
        return keras.Sequential([conv, bn, leaky_relu])
    else:
        return keras.Sequential([conv, bn])
