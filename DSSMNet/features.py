import tensorflow as tf
from basics import *


class ResBlock(keras.Model):
    """
    Residual block, no ReLU after summation.
    """
    def __init__(self, filters, dilation_rate):
        super(ResBlock, self).__init__()

        self.conv1 = conv2d_bn_act(filters, 3, 1, 'same', dilation_rate, 'relu')
        self.conv2 = conv2d_bn_act(filters, 3, 1, 'same', dilation_rate, None)

    def call(self, inputs, training=None, mask=None):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs += inputs

        return outputs


def make_layers(filters, dilation_rate, num):
    """
    Build sequential residual block.
    """
    block = keras.Sequential()
    for i in range(num):
        block.add(ResBlock(filters, dilation_rate))

    return block


def pool(pool_size, strides, filters):
    """
    Average Pooling followed by convolutions.
    """
    pooling = keras.layers.AvgPool2D(pool_size, strides)
    conv1 = conv2d_bn_act(filters, 3, 1, 'same', 1, 'relu')

    return keras.Sequential([pooling, conv1])


class Feature(keras.Model):
    def __init__(self, filters):
        super(Feature, self).__init__()

        self.conv1 = conv2d_bn_act(filters, 5, 2, 'same', 1, 'relu')
        self.conv2 = conv2d_bn_act(filters, 3, 1, 'same', 1, 'relu')
        self.res1 = make_layers(filters, 1, 6)
        self.res2 = make_layers(filters, 2, 2)
        self.res3 = make_layers(filters, 4, 2)
        self.pooling1 = pool(2, 2, filters)
        self.pooling2 = pool(4, 4, filters)
        self.res4 = make_layers(filters, 1, 3)
        self.res5 = make_layers(filters, 1, 3)
        self.conv3 = conv2d(filters, 1, 1, 'valid', 1, False)
        self.conv4 = conv2d(filters, 1, 1, 'valid', 1, False)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x1 = self.pooling1(x)
        x2 = self.pooling2(x)
        x1 = self.res4(x1)
        x2 = self.res5(x2)
        x1 = self.conv3(x1)
        x2 = self.conv4(x2)

        return [x1, x2]     # 1/4, 1/8


class FasterFeature(keras.Model):
    def __init__(self, filters):
        super(FasterFeature, self).__init__()

        self.conv1 = conv2d_bn_act(filters, 5, 2, 'same', 1, 'relu')
        self.conv2 = conv2d_bn_act(filters, 3, 1, 'same', 1, 'relu')
        self.res1 = make_layers(filters, 1, 3)
        self.res2 = make_layers(filters, 2, 2)
        self.res3 = make_layers(filters, 4, 2)
        self.pooling1 = pool(2, 2, filters)
        self.pooling2 = pool(4, 4, filters)
        self.res4 = make_layers(filters, 1, 6)
        self.res5 = make_layers(filters, 1, 6)
        self.conv3 = conv2d(filters, 1, 1, 'valid', 1, False)
        self.conv4 = conv2d(filters, 1, 1, 'valid', 1, False)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x1 = self.pooling1(x)
        x2 = self.pooling2(x)
        x1 = self.res4(x1)
        x2 = self.res5(x2)
        x1 = self.conv3(x1)
        x2 = self.conv4(x2)

        return [x1, x2]     # 1/4, 1/8


class HighFeature(keras.Model):
    def __init__(self, filters):
        super(HighFeature, self).__init__()

        self.conv1 = conv2d_bn_act(filters, 5, 2, 'same', 1, 'relu')
        self.conv2 = conv2d_bn_act(filters, 3, 1, 'same', 1, 'relu')
        self.res1 = make_layers(filters, 1, 6)
        self.res2 = make_layers(filters, 2, 2)
        self.res3 = make_layers(filters, 4, 2)
        self.pooling1 = pool(2, 2, filters)
        self.res4 = make_layers(filters, 1, 3)
        self.conv3 = conv2d(filters, 1, 1, 'valid', 1, False)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pooling1(x)
        x = self.res4(x)
        x = self.conv3(x)

        return x      # 1/4


class LowFeature(keras.Model):
    def __init__(self, filters):
        super(LowFeature, self).__init__()

        self.conv1 = conv2d_bn_act(filters, 5, 2, 'same', 1, 'relu')
        self.conv2 = conv2d_bn_act(filters, 3, 1, 'same', 1, 'relu')
        self.res1 = make_layers(filters, 1, 6)
        self.res2 = make_layers(filters, 2, 2)
        self.res3 = make_layers(filters, 4, 2)
        self.pooling2 = pool(4, 4, filters)
        self.res5 = make_layers(filters, 1, 3)
        self.conv4 = conv2d(filters, 1, 1, 'valid', 1, False)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pooling2(x)
        x = self.res5(x)
        x = self.conv4(x)

        return x


class FeatureVolume(keras.Model):
    """
    Build feature volume by concatenation or difference operation.
    """
    def __init__(self, max_disp, mode='concat'):
        # mode: concat or diff
        super(FeatureVolume, self).__init__()

        self.max_disp = max_disp
        self.mode = mode

    def call(self, inputs, training=None, mask=None):
        assert len(inputs) == 2
        volume = []

        if self.mode == 'concat':
            for i in range(-self.max_disp, self.max_disp):
                if i < 0:
                    volume.append(tf.pad(
                        tensor=tf.concat([inputs[0][:, :, :i, :], inputs[1][:, :, -i:, :]], -1),
                        paddings=[[0, 0], [0, 0], [0, -i], [0, 0]], mode='CONSTANT'))
                elif i > 0:
                    volume.append(tf.pad(
                        tensor=tf.concat([inputs[0][:, :, i:, :], inputs[1][:, :, :-i, :]], -1),
                        paddings=[[0, 0], [0, 0], [i, 0], [0, 0]], mode='CONSTANT'))
                else:
                    volume.append(tf.concat([inputs[0], inputs[1]], -1))

        elif self.mode == 'diff':
            for i in range(-self.max_disp, self.max_disp):
                if i < 0:
                    volume.append(tf.pad(
                        tensor=inputs[0][:, :, :i, :] - inputs[1][:, :, -i:, :],
                        paddings=[[0, 0], [0, 0], [0, -i], [0, 0]], mode='CONSTANT'))
                elif i > 0:
                    volume.append(tf.pad(
                        tensor=inputs[0][:, :, i:, :] - inputs[1][:, :, :-i, :],
                        paddings=[[0, 0], [0, 0], [i, 0], [0, 0]], mode='CONSTANT'))
                else:
                    volume.append(inputs[0] - inputs[1])

        else:
            raise Exception('Must be concat or diff!')

        return tf.stack(volume, 1)     # [N, D, H, W, C]
