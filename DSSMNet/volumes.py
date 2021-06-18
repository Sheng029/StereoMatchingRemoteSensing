import tensorflow as tf
from basics import *


class CostVolumeAggregation(keras.Model):
    def __init__(self, filters):
        super(CostVolumeAggregation, self).__init__()

        self.conv1 = conv3d_bn_act(filters, 3, 1, 'same', 'relu')
        self.conv2 = conv3d_bn_act(2 * filters, 3, 2, 'same', 'relu')
        self.conv3 = conv3d_bn_act(2 * filters, 3, 1, 'same', 'relu')
        self.conv4 = conv3d_bn_act(4 * filters, 3, 2, 'same', 'relu')
        self.conv5 = conv3d_bn_act(4 * filters, 3, 1, 'same', 'relu')
        self.conv6 = conv3d_transpose_bn_act(2 * filters, 3, 2, 'same', 'relu')
        self.conv7 = conv3d_bn_act(2 * filters, 3, 1, 'same', 'relu')
        self.conv8 = conv3d_transpose_bn_act(filters, 3, 2, 'same', 'relu')
        self.conv9 = conv3d_bn_act(filters, 3, 1, 'same', 'relu')
        self.conv10 = conv3d(1, 1, 1, 'same', False)

    def call(self, inputs, training=None, mask=None):
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x2 = self.conv3(x2)
        x3 = self.conv4(x2)
        x3 = self.conv5(x3)
        x4 = self.conv6(x3)
        x4 += x2
        x4 = self.conv7(x4)
        x5 = self.conv8(x4)
        x5 += x1
        x5 = self.conv9(x5)
        outputs = self.conv10(x5)

        return outputs  # [N, D, H, W, 1]


class CostVolume(keras.Model):
    """
    Cost volume computation.
    """
    def __init__(self, filters):
        super(CostVolume, self).__init__()

        self.conv1 = conv3d_bn_act(filters, 3, 1, 'same', 'relu')
        self.conv2 = conv3d_bn_act(2*filters, 3, 2, 'same', 'relu')
        self.conv3 = conv3d_bn_act(2*filters, 3, 1, 'same', 'relu')
        self.conv4 = conv3d_bn_act(4*filters, 3, 2, 'same', 'relu')
        self.conv5 = conv3d_bn_act(4*filters, 3, 1, 'same', 'relu')
        self.conv6 = conv3d_transpose_bn_act(2*filters, 3, 2, 'same', 'relu')
        self.conv7 = conv3d_bn_act(2*filters, 3, 1, 'same', 'relu')
        self.conv8 = conv3d_transpose_bn_act(filters, 3, 2, 'same', 'relu')
        self.conv9 = conv3d_bn_act(filters, 3, 1, 'same', 'relu')
        self.conv10 = conv3d(filters, 1, 1, 'same', False)

    def call(self, inputs, training=None, mask=None):
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x2 = self.conv3(x2)
        x3 = self.conv4(x2)
        x3 = self.conv5(x3)
        x4 = self.conv6(x3)
        x4 += x2
        x4 = self.conv7(x4)
        x5 = self.conv8(x4)
        x5 += x1
        x5 = self.conv9(x5)
        outputs = self.conv10(x5)

        return outputs      # [N, D, H, W, C]


class Aggregation(keras.Model):
    """
    Cost volume aggreation.
    """
    def __init__(self, filters):
        super(Aggregation, self).__init__()

        self.agg1 = keras.Sequential([conv3d_bn_act(filters, (3, 1, 1), 1, 'same', 'leaky_relu'),
                                      conv3d_bn_act(filters, (1, 3, 1), 1, 'same', 'leaky_relu'),
                                      conv3d_bn_act(filters, (1, 1, 3), 1, 'same', 'leaky_relu')])
        self.agg2 = keras.Sequential([conv3d_bn_act(filters, (3, 1, 1), 1, 'same', 'leaky_relu'),
                                      conv3d_bn_act(filters, (1, 3, 1), 1, 'same', 'leaky_relu'),
                                      conv3d_bn_act(filters, (1, 1, 3), 1, 'same', 'leaky_relu')])
        self.conv = conv3d(1, 1, 1, 'valid', False)

    def call(self, inputs, training=None, mask=None):
        outputs = self.agg1(inputs)
        outputs = self.agg2(outputs)
        outputs = self.conv(outputs)

        return outputs     # [N, D, H, W, 1]


class SimpleAggregation(keras.Model):
    def __init__(self, filters):
        super(SimpleAggregation, self).__init__()

        self.conv1 = conv3d_bn_act(filters, 3, 1, 'same', 'relu')
        self.conv2 = conv3d_bn_act(filters, 3, 1, 'same', 'relu')
        self.conv3 = conv3d_bn_act(filters, 3, 1, 'same', 'relu')
        self.conv4 = conv3d_bn_act(filters, 3, 1, 'same', 'relu')
        self.conv5 = conv3d_bn_act(filters, 3, 1, 'same', 'relu')
        self.conv6 = conv3d_bn_act(filters, 3, 1, 'same', 'leaky_relu')
        self.conv7 = conv3d_bn_act(filters, 3, 1, 'same', 'leaky_relu')
        self.conv8 = conv3d(1, 1, 1, 'valid', False)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        return x


def change_dim(inputs):
    """
    Squeeze and transpose [N, D, H, W, 1] to [N, H, W, D].
    """
    outputs = tf.squeeze(inputs, -1)
    outputs = tf.transpose(outputs, (0, 2, 3, 1))

    return outputs      # [N, H, W, D]
