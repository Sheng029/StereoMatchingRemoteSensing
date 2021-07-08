import tensorflow as tf
import tensorflow.keras as keras
from aggregations import conv3d


class DisparityComputation(keras.Model):
    def __init__(self, max_disp):
        super(DisparityComputation, self).__init__()
        self.max_disp = max_disp
        self.conv = conv3d(1, (1, 1, 1), (1, 1, 1), 'valid', False)

    def call(self, inputs, training=None, mask=None):
        # inputs: [N, D, H, W, C]
        assert inputs.shape[1] == 2 * self.max_disp

        cost_volume = self.conv(inputs)     # [N, D, H, W, 1]
        cost_volume = tf.squeeze(cost_volume, -1)     # [N, D, H, W]
        cost_volume = tf.transpose(cost_volume, (0, 2, 3, 1))   # [N, H, W, D]

        candidates = tf.linspace(-1.0 * self.max_disp, 1.0 * self.max_disp - 1.0, 2 * self.max_disp)
        probabilities = tf.math.softmax(-1.0 * cost_volume, -1)
        disparity = tf.reduce_sum(candidates * probabilities, -1, True)

        return disparity     # [N, H, W, 1]

