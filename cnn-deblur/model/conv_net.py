import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Layer, Input, Conv2D, Conv2DTranspose, Activation, Add)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_absolute_error
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import plot_model
from typing import List, Tuple, Optional


def ssim_loss(trueY, predY):
    return -tf.image.ssim(trueY, predY, max_val=2.)


def mix_loss(trueY, predY):
    alpha = 0.84
    return alpha * ssim_loss(trueY, predY) + (1 - alpha) * tf.math.reduce_mean(mean_absolute_error(trueY, predY))


def content_loss(trueY, predY):
    return 0.5 * K.sum(K.square(trueY - predY))


def ResConv(kernels: List[int],
            filters_num: List[int],
            res_in: Layer,
            layer_idx: int,
            blocks_num: Optional[int] = 1,
            double_first_stride: Optional[bool] = False,
            use_res_conv: Optional[bool] = False,
            res_filter: Optional[int] = None,
            res_size: Optional[int] = None,
            res_stride: Optional[int] = None):
    x = res_in

    # If 'use_res_conv' is set, apply a convolution on the residual path instead of an identity
    if use_res_conv:
        res_in = Conv2D(res_filter, kernel_size=res_size, strides=res_stride, padding='same',
                        name='conv{0:d}_shortcut'.format(layer_idx))(res_in)

    for n in range(1, blocks_num + 1):
        n_sub = 1
        for kernel, fltr in zip(kernels, filters_num):
            # Update the suffix of layer's name
            layer_suffix = '{0:d}_{1:d}.{2:d}'.format(layer_idx, n, n_sub)

            # Check if the first stride must be doubled
            if double_first_stride and n == 1 and n_sub == 1:
                stride = 2
            else:
                stride = 1

            x = Conv2D(fltr, kernel_size=kernel, strides=stride, padding='same', activation='relu',
                       name='conv{0:s}'.format(layer_suffix))(x)
            # x = BatchNormalization(name='bn{0:s}'.format(layer_suffix))(x)

            # Update sub-block counter
            n_sub += 1

    # Add the residual path to the main one
    x = Add()([x, res_in])
    x = Activation('relu', name='relu{0:d}'.format(layer_idx))(x)
    return x


def ResConvTranspose(kernels: List[int],
                     filters_num: List[int],
                     res_in: Layer,
                     layer_idx: int,
                     blocks_num: Optional[int] = 1,
                     double_last_stride: Optional[bool] = False,
                     use_res_tconv: Optional[bool] = False,
                     res_filter: Optional[int] = None,
                     res_size: Optional[int] = None,
                     res_stride: Optional[int] = None):
    x = res_in
    # If 'use_res_tconv' is set, apply a convolution on the residual path instead of an identity
    if use_res_tconv:
        res_in = Conv2DTranspose(res_filter, kernel_size=res_size, strides=res_stride, padding='same',
                                 name='t_conv{0:d}_shortcut'.format(layer_idx))(res_in)

    for n in range(1, blocks_num + 1):
        n_sub = 1
        for kernel, fltr in zip(kernels, filters_num):
            # Update the suffix of layer's name
            layer_suffix = '{0:d}_{1:d}.{2:d}'.format(layer_idx, n, n_sub)

            # Check if the last stride must be doubled
            if double_last_stride and n == blocks_num and n_sub == len(kernels):
                stride = 2
            else:
                stride = 1

            x = Conv2DTranspose(fltr, kernel_size=kernel, strides=stride, padding='same', activation='relu',
                                name='t_conv{0:s}'.format(layer_suffix))(x)
            # x = BatchNormalization(name='bn{0:s}'.format(layer_suffix))(x)

            # Update sub-block counter
            n_sub += 1

    # Add the residual path to the main one
    x = Add()([x, res_in])
    x = Activation('relu', name='relu{0:d}'.format(layer_idx))(x)
    return x


class ConvNet:
    """Class implementing a ResNet architecture"""

    def __init__(self, input_shape: Tuple[int, int, int]):
        # ENCODER
        visible = Input(shape=input_shape)
        # Convolution layer: Conv + ReLU + BatchNorm
        conv = Conv2D(16,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      activation='relu',
                      name='conv')(visible)
        # conv = BatchNormalization(axis=3, name='bn')(conv)
        # First layer: 2x(Conv + ReLU) + Identity Residual (16 filters)
        layer1 = ResConv(kernels=[3, 3],
                         filters_num=[16, 16],
                         res_in=conv,
                         layer_idx=1)
        # Second layer: 2x(Conv + ReLU) with double first stride + Conv Residual (32 filters)
        layer2 = ResConv(kernels=[3, 3],
                         filters_num=[32, 32],
                         res_in=layer1,
                         layer_idx=2,
                         double_first_stride=True,
                         use_res_conv=True,
                         res_filter=32,
                         res_size=3,
                         res_stride=2)
        # Third layer: same as second layer, but with 64 filters
        layer3 = ResConv(kernels=[3, 3],
                         filters_num=[64, 64],
                         res_in=layer2,
                         layer_idx=3,
                         double_first_stride=True,
                         use_res_conv=True,
                         res_filter=64,
                         res_size=3,
                         res_stride=2)
        # Forth layer: same as third layer, but with 128 filters
        layer4 = ResConv(kernels=[3, 3],
                         filters_num=[128, 128],
                         res_in=layer3,
                         layer_idx=4,
                         double_first_stride=True,
                         use_res_conv=True,
                         res_filter=128,
                         res_size=3,
                         res_stride=2)
        # DECODER
        # Fifth layer: 2x(DeConv + ReLU) with double last stride + Conv Residual (128 filters)
        layer5 = ResConvTranspose(kernels=[3, 3],
                                  filters_num=[128, 64],
                                  res_in=layer4,
                                  layer_idx=5,
                                  double_last_stride=True,
                                  use_res_tconv=True,
                                  res_filter=64,
                                  res_size=3,
                                  res_stride=2)
        # Sixth layer: same as fifth layer, but with 64 filters
        layer6 = ResConvTranspose(kernels=[3, 3],
                                  filters_num=[64, 32],
                                  res_in=layer5,
                                  layer_idx=6,
                                  double_last_stride=True,
                                  use_res_tconv=True,
                                  res_filter=32,
                                  res_size=3,
                                  res_stride=2)
        # Seventh layer: same as sixth layer, but with 32 filters
        layer7 = ResConvTranspose(kernels=[3, 3],
                                  filters_num=[32, 16],
                                  res_in=layer6,
                                  layer_idx=7,
                                  double_last_stride=True,
                                  use_res_tconv=True,
                                  res_filter=16,
                                  res_size=3,
                                  res_stride=2)
        # Eighth layer: 2x(DeConv + ReLU) + Identity Residual (16 filters)
        layer8 = ResConvTranspose(kernels=[3, 3],
                                  filters_num=[16, 16],
                                  res_in=layer7,
                                  layer_idx=8)
        # Convolution layer: Conv + ReLU
        tconv = Conv2DTranspose(3, kernel_size=3, padding='same', activation='relu')(layer8)

        self.model = Model(inputs=visible, outputs=tconv)
        self.model.compile(Adam(learning_rate=1e-4), loss=ssim_loss, metrics=['accuracy'])

    def fit(self,
            x,
            y,
            batch_size: int,
            epochs: int,
            steps_per_epoch: Optional[int] = None,
            validation_data: Optional = None,
            initial_epoch: Optional[int] = 0,
            callbacks: Optional[List[Callback]] = None):
        return self.model.fit(x, y,
                              batch_size=batch_size,
                              epochs=epochs,
                              steps_per_epoch=steps_per_epoch,
                              validation_data=validation_data,
                              initial_epoch=initial_epoch,
                              callbacks=callbacks)

    def evaluate(self,
                 testX,
                 testY,
                 batch_size: int):
        self.model.evaluate(testX, testY, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)

    def summary(self):
        self.model.summary()

    def plot_model(self, path):
        plot_model(self.model, to_file=path, show_shapes=True)

model = ConvNet((32, 32, 3))
print(model.summary())
