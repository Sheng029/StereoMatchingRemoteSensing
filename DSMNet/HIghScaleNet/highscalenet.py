import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image
from features import HighScaleFeatureExtractor
from costs import CostDifference
from aggregations import FactorizedCostAggregation
from computation import DisparityComputation
from refinement import Refinement
from utils.data_loader import read_tif
from utils.metric import epe_metric, d1_metric


class HighScaleModel:
    def __init__(self, height=1024, width=1024, channel=3, max_disp=64):
        self.height = height
        self.width = width
        self.channel = channel
        self.max_disp = max_disp
        self.model = None

    def build_model(self):
        left_image = keras.Input(shape=(self.height, self.width, self.channel))
        right_image = keras.Input(shape=(self.height, self.width, self.channel))

        feature_extractor = HighScaleFeatureExtractor(filters=16)
        left_feature = feature_extractor(left_image)
        right_feature = feature_extractor(right_image)

        cost_difference = CostDifference(max_disp=self.max_disp // 4)
        cost_volume = cost_difference([left_feature, right_feature])

        aggregation = FactorizedCostAggregation(filters=16)
        agg_cost_volume = aggregation(cost_volume)

        computation = DisparityComputation(max_disp=self.max_disp // 4)
        disparity = computation(agg_cost_volume)  # 1/4

        refine = Refinement(filters=16)
        refined_disparity = refine([disparity, left_image])  # 1/2

        d0 = tf.image.resize(disparity, [self.height, self.width]) * 4
        d1 = tf.image.resize(refined_disparity, [self.height, self.width]) * 2

        self.model = keras.Model(inputs=[left_image, right_image], outputs=[d0, d1])
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
            disparity = self.model.predict([left_image, right_image])   # [d0, d1]
            disparity = Image.fromarray(disparity[-1][0, :, :, 0])
            name = left.replace('RGB', 'DSP')
            disparity.save(os.path.join(output_dir, name))


if __name__ == '__main__':
    # predict
    left_dir = '../../examples/left'
    right_dir = '../../examples/right'
    output_dir = 'prediction'
    weights = 'HighScaleModel.h5'

    model = HighScaleModel()
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
