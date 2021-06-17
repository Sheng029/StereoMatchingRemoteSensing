import tensorflow as tf
import tensorflow.keras as keras
from basics import conv2d, conv2d_bn_act


class Computation(keras.Model):
    """
    Soft ArgMin.
    """
    def __init__(self, max_disp):
        super(Computation, self).__init__()
        self.max_disp = max_disp

    def call(self, inputs, training=None, mask=None):
        # inputs: [N, H, W, D], D == 2 * max_disp
        assert inputs.shape[-1] == 2 * self.max_disp

        disp_candidates = tf.linspace(-1.0 * self.max_disp, 1.0 * self.max_disp - 1.0, 2 * self.max_disp)
        prob_volume = tf.math.softmax(-1.0 * inputs, -1)
        disparity = tf.reduce_sum(disp_candidates * prob_volume, -1, True)

        return disparity        # [N, H, W, 1]


class Refinement(keras.Model):
    def __init__(self, filters):
        super(Refinement, self).__init__()

        self.conv1 = conv2d_bn_act(filters, 3, 1, 'same', 1, 'leaky_relu')
        self.conv2 = conv2d_bn_act(filters, 3, 1, 'same', 1, 'leaky_relu')
        self.conv3 = conv2d_bn_act(filters, 3, 1, 'same', 2, 'leaky_relu')
        self.conv4 = conv2d_bn_act(filters, 3, 1, 'same', 4, 'leaky_relu')
        self.conv5 = conv2d_bn_act(filters, 3, 1, 'same', 1, 'leaky_relu')
        self.conv6 = conv2d(1, 3, 1, 'same', 1, False)

    def call(self, inputs, training=None, mask=None):
        # inputs: [low disparity, resized left image], [(N, H, W, 1), ([N, H, W, 3)])
        assert len(inputs) == 2

        # Up-sample low disparity to corresponding resolution as the resized image
        scale_factor = inputs[1].shape[1] / inputs[0].shape[1]
        disp = tf.image.resize(images=inputs[0], size=[inputs[1].shape[1], inputs[1].shape[2]])
        disp *= scale_factor

        # Concat the disparity and image
        concat = tf.concat([disp, inputs[1]], -1)
        # learn residual signal
        x = self.conv1(concat)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        outputs = disp + x

        return outputs
