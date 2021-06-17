import os
import numpy as np
import tensorflow.keras as keras
from PIL import Image
from tensorflow.keras import layers, models
from Utils.data_loader import read_tif


class DenseMapNet:
    def __init__(self, height=1024, width=1024, channel=3, max_disp=64):
        self.height = height
        self.width = width
        self.channel = channel
        self.max_disp = max_disp
        self.model = None

    def build_model(self, dropout=0.2):
        left = keras.Input(shape=(self.height, self.width, self.channel))
        right = keras.Input(shape=(self.height, self.width, self.channel))

        # left image as reference
        x = layers.Conv2D(filters=16, kernel_size=5, padding='same')(left)
        xleft = layers.Conv2D(filters=1, kernel_size=5, padding='same', dilation_rate=2)(left)

        # left and right images for disparity estimation
        xin = layers.Concatenate()([left, right])
        xin = layers.Conv2D(filters=32, kernel_size=5, padding='same')(xin)

        # image reduced by 8
        x8 = layers.MaxPooling2D(8)(xin)
        x8 = layers.BatchNormalization()(x8)
        x8 = layers.Activation('relu')(x8)

        dilation_rate = 1
        y = x8
        # correspondence network
        # parallel cnn at increasing dilation rate
        for i in range(4):
            a = layers.Conv2D(filters=32, kernel_size=5, padding='same', dilation_rate=dilation_rate)(x8)
            a = layers.Dropout(dropout)(a)
            y = layers.Concatenate()([a, y])
            dilation_rate += 1

        dilation_rate = 1
        x = layers.MaxPooling2D(8)(x)
        # disparity network
        # dense interconnection inspired by DenseNet
        for i in range(4):
            x = layers.Concatenate()([x, y])
            y = layers.BatchNormalization()(x)
            y = layers.Activation('relu')(y)
            y = layers.Conv2D(filters=64, kernel_size=1, padding='same')(y)

            y = layers.BatchNormalization()(y)
            y = layers.Activation('relu')(y)
            y = layers.Conv2D(filters=16, kernel_size=5, padding='same', dilation_rate=dilation_rate)(y)
            y = layers.Dropout(dropout)(y)
            dilation_rate += 1

        # disparity estimate scaled back to original image size
        x = layers.Concatenate()([x, y])
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters=32, kernel_size=1, padding='same')(x)
        x = layers.UpSampling2D(8)(x)
        # if not self.settings.nopadding:
        #     x = layers.ZeroPadding2D(padding=(2, 0))(x)

        # left image skip connection to disparity estimate
        x = layers.Concatenate()([x, xleft])
        y = layers.BatchNormalization()(x)
        y = layers.Activation('relu')(y)
        y = layers.Conv2D(filters=16, kernel_size=5, padding='same')(y)

        x = layers.Concatenate()([x, y])
        y = layers.BatchNormalization()(x)
        y = layers.Activation('relu')(y)
        yout = layers.Conv2DTranspose(filters=1, kernel_size=9, padding='same')(y)

        # densemapnet model
        self.model = models.Model(inputs=[left, right], outputs=yout)
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
            disparity = self.model.predict([left_image, right_image])
            disparity = Image.fromarray(disparity[0, :, :, 0])
            name = left.replace('RGB', 'DSP')
            disparity.save(os.path.join(output_dir, name))


if __name__ == '__main__':
    # predict
    left_dir = '../Examples/OMA/Left'
    right_dir = '../Examples/OMA/Right'
    output_dir = 'prediction/OMA'
    weights = 'DenseMapNet.h5'

    net = DenseMapNet(1024, 1024, 3, 64)
    net.build_model()
    net.predict(left_dir, right_dir, output_dir, weights)
