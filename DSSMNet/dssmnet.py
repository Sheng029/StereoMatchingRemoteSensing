import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image
from features import Feature, FeatureVolume
from volumes import CostVolume, Aggregation, change_dim
from disparities import Computation, Refinement
from Utils.data_loader import read_tif


class DSSMNet:
    def __init__(self, height=1024, width=1024, channel=3, max_disp=64):
        self.height = height
        self.width = width
        self.channel = channel
        self.max_disp = max_disp
        self.model = None

    def build_model(self):
        # inputs, [None, 1024, 1024, 3]
        left_image = keras.Input(shape=(self.height, self.width, self.channel))
        right_image = keras.Input(shape=(self.height, self.width, self.channel))

        # feature extraction, [(N, 256, 256, 32), (N, 128, 128, 32)]
        feature_extractor = Feature(32)
        [left_high, left_low] = feature_extractor(left_image)
        [right_high, right_low] = feature_extractor(right_image)

        # feature volumes, [(N, 32, 256, 256, 32), (N, 16, 128, 128, 32)]
        high_constructor = FeatureVolume(self.max_disp // 4, 'diff')
        low_constructor = FeatureVolume(self.max_disp // 8, 'diff')
        high_volume = high_constructor([left_high, right_high])
        low_volume = low_constructor([left_low, right_low])

        # Cost volumes, [(N, 32, 256, 256, 16), (N, 16, 128, 128, 16)]
        high_similarity = CostVolume(16)
        low_similarity = CostVolume(16)
        high_cost_volume = high_similarity(high_volume)
        low_cost_volume = low_similarity(low_volume)

        # volume aggregation, [N, 32, 256, 256, 1], [N, 16, 128, 128, 1]
        high_aggregator = Aggregation(16)
        low_aggregator = Aggregation(16)
        aggregated_high = high_aggregator(high_cost_volume)
        aggregated_low = low_aggregator(low_cost_volume)

        # low disparity computation
        low_computer = Computation(self.max_disp // 8)
        low_disp = low_computer(change_dim(aggregated_low))  # 1/8 disparity

        # high disparity computation
        up = keras.layers.UpSampling3D((2, 2, 2))
        high_computer = Computation(self.max_disp // 4)
        low_to_high = up(aggregated_low)
        aggregated_high += low_to_high
        high_disp = high_computer(change_dim(aggregated_high))  # 1/4 disparity

        # refinement, 1/2 disparity
        refiner = Refinement(16)
        left_2x = tf.image.resize(left_image, [self.height // 2, self.width // 2])
        refined_disp = refiner([high_disp, left_2x])

        # multi outputs
        d0 = tf.image.resize(low_disp, [self.height, self.width]) * 8
        d1 = tf.image.resize(high_disp, [self.height, self.width]) * 4
        d2 = tf.image.resize(refined_disp, [self.height, self.width]) * 2

        # model
        self.model = keras.Model(inputs=[left_image, right_image], outputs=[d0, d1, d2])
        self.model.summary()

    def predict(self, left_dir, right_dir, output_dir, weights):
        self.model.load_weights(weights)

        lefts = os.listdir(left_dir)
        rights = os.listdir(right_dir)
        lefts.sort()
        rights.sort()
        assert len(lefts) == len(rights)

        for left, right in zip(lefts, rights):
            left_image = np.expand_dims(read_tif(os.path.join(left_dir, left)), 0)
            right_image = np.expand_dims(read_tif(os.path.join(right_dir, right)), 0)
            disp = self.model.predict([left_image, right_image])   # [d0, d1, d2]
            disp = Image.fromarray(disp[-1][0, :, :, 0])
            name = left.replace('RGB', 'DSP')
            disp.save(os.path.join(output_dir, name))


if __name__ == '__main__':
    # predict
    left_dir = '../Examples/JAX/Left'
    right_dir = '../Examples/JAX/Right'
    output_dir = 'prediction/JAX'
    weights = 'DSSMNet.h5'
    net = DSSMNet(1024, 1024, 3, 64)
    net.build_model()
    net.predict(left_dir, right_dir, output_dir, weights)
