import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image
from features import FeatureExtractor
from costs import CostDifference
from aggregations import FactorizedCostAggregation
from computation import DisparityComputation
from refinement import Refinement
from utils.data_loader import read_tif
from utils.metric import epe_metric, d1_metric


class FullModel:
    def __init__(self, height=1024, width=1024, channel=3, max_disp=64):
        self.height = height
        self.width = width
        self.channel = channel
        self.max_disp = max_disp
        self.model = None

    def build_model(self):
        # inputs [H, W, C], [1024, 1024, 3]
        left_image = keras.Input(shape=(self.height, self.width, self.channel))
        right_image = keras.Input(shape=(self.height, self.width, self.channel))

        # features [H, W, C], [256, 256, 16], [128, 128, 16]
        feature_extractor = FeatureExtractor(filters=16)
        [left_high_feature, left_low_feature] = feature_extractor(left_image)
        [right_high_feature, right_low_feature] = feature_extractor(right_image)

        # cost volume [D, H, W, C], [32, 256, 256, 16], [16, 128, 128, 16]
        high_cost_difference = CostDifference(max_disp=self.max_disp // 4)
        low_cost_difference = CostDifference(max_disp=self.max_disp // 8)
        high_cost_volume = high_cost_difference([left_high_feature, right_high_feature])
        low_cost_volume = low_cost_difference([left_low_feature, right_low_feature])

        # low cost volume aggregation [D, H, W, C], [16, 128, 128, 16]
        low_aggregation = FactorizedCostAggregation(filters=16)
        low_agg_cost_volume = low_aggregation(low_cost_volume)

        # low disparity computation [H, W, 1], [128, 128, 1]
        low_computation = DisparityComputation(max_disp=self.max_disp // 8)
        low_disparity = low_computation(low_agg_cost_volume)   # 1/8

        # high cost volume aggregation [D, H, W, C], [32, 256, 256, 16]
        # firstly, up-sample aggregated low cost volume and add it to high cost volume
        upsample = keras.layers.UpSampling3D(size=(2, 2, 2))
        low_to_high = upsample(low_agg_cost_volume)
        high_cost_volume += low_to_high
        # secondly, aggregate
        high_aggregation = FactorizedCostAggregation(filters=16)
        high_agg_cost_volume = high_aggregation(high_cost_volume)

        # high disparity computation [H, W, 1], [256, 256, 1]
        high_computation = DisparityComputation(max_disp=self.max_disp // 4)
        high_disparity = high_computation(high_agg_cost_volume)   # 1/4

        # refinement [H, W, 1], [512, 512, 1]
        refine = Refinement(filters=16)
        refined_disparity = refine([high_disparity, left_image])   # 1/2

        # predicted disparity maps are bilinearly up-sampled to match the ground-truth resolution
        d0 = tf.image.resize(low_disparity, [self.height, self.width]) * 8
        d1 = tf.image.resize(high_disparity, [self.height, self.width]) * 4
        d2 = tf.image.resize(refined_disparity, [self.height, self.width]) * 2

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
            disparity = self.model.predict([left_image, right_image])   # [d0, d1, d2]
            disparity = Image.fromarray(disparity[-1][0, :, :, 0])
            name = left.replace('RGB', 'DSP')
            disparity.save(os.path.join(output_dir, name))


if __name__ == '__main__':
    # predict
    left_dir = '../../examples/left'
    right_dir = '../../examples/right'
    output_dir = 'prediction'
    weights = 'FullModel.h5'

    model = FullModel()
    model.build_model()
    model.predict(left_dir, right_dir, output_dir, weights)

    # evaluate
    est_dir = 'prediction'
    gt_dir = '../../examples/disp'
    ests = os.listdir(est_dir)
    gts = os.listdir(gt_dir)
    ests.sort()
    gts.sort()

    for est, gt in zip(ests, gts):
        epe = epe_metric(os.path.join(est_dir, est), os.path.join(gt_dir, gt), 64.0)[-1]
        d1 = d1_metric(os.path.join(est_dir, est), os.path.join(gt_dir, gt), 64.0)[-1]
        print('Tile: %s, EPE: %f, D1: %f' % (est[0:-9], epe, d1))
