from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Layer, Input, Conv2D, Conv2DTranspose, BatchNormalization,
                                     Activation, Add, MaxPooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.utils import plot_model
import numpy as np
from typing import List, Tuple, Optional


def add_res_layer(kernels: List[int],
                  filters_num: List[int],
                  res_in: Layer,
                  layer_idx: int,
                  blocks_num: Optional[int] = 1,
                  double_first_stride: Optional[bool] = False,
                  use_res_conv: Optional[bool] = False,
                  res_filter: Optional[int] = None,
                  res_size: Optional[int] = None,
                  res_stride: Optional[int] = None,
                  use_maxpool: Optional[bool] = False,
                  pool_size: Optional[int] = None,
                  pool_stride: Optional[int] = None):
    # If 'use_maxpool' is set, apply a MaxPooling with specified parameters as first layer of the block
    if use_maxpool:
        x = MaxPooling2D(pool_size=pool_size, strides=pool_stride, name='pool{0:d}'.format(layer_idx))
    else:
        x = res_in

    # If 'res_conv' is set, apply a convolution on the residual path instead of an identity
    if use_res_conv:
        res_in = Conv2D(res_filter, kernel_size=res_size, strides=res_stride, padding='same')(res_in)

    for n in range(1, blocks_num + 1):
        n_sub = 1
        for kernel, fltr in zip(kernels, filters_num):
            # Update the suffix of layer's name
            layer_suffix = '{0:d}_{1:d}.{2:d}'.format(layer_idx, n, n_sub)

            # Check if the first stride must be doubled
            if double_first_stride and n * n_sub == 1:
                stride = 2
            else:
                stride = 1

            x = Conv2D(fltr, kernel_size=kernel, strides=stride, padding='same',
                       name='conv{0:s}'.format(layer_suffix))(x)
            x = BatchNormalization(axis=3, name='bn{0:s}'.format(layer_suffix))(x)
            x = Activation('relu', name='relu{0:s}'.format(layer_suffix))(x)

            # Update sub-block counter
            n_sub += 1

    # Add the residual path to the main one
    x = Add()([x, res_in])
    x = Activation('relu', name='relu{0:d}'.format(layer_idx))(x)
    return x


class ConvNet:
    """Class implementing a ResNet architecture"""

    def __init__(self, input_shape: Tuple[int, int, int]):
        visible = Input(shape=input_shape)
        # Convolution layer: Conv + BatchNorm + ReLU
        conv = Conv2D(16,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      name='conv')(visible)
        conv = BatchNormalization(axis=3, name='bn')(conv)
        conv = Activation('relu', name='relu')(conv)
        # First layer: 2x(Conv + BatchNorm) + Identity Residual (16 filters)
        layer1 = add_res_layer(kernels=[3, 3],
                               filters_num=[16, 16],
                               res_in=conv,
                               layer_idx=1)
        # Second layer: 2x(Conv + BatchNorm) which double first stride + Conv Residual (32 filters)
        layer2 = add_res_layer(kernels=[3, 3],
                               filters_num=[32, 32],
                               res_in=layer1,
                               layer_idx=2,
                               double_first_stride=True,
                               use_res_conv=True,
                               res_filter=32,
                               res_size=3,
                               res_stride=2)
        # Third layer: same as second layer, but with 64 filters
        layer3 = add_res_layer(kernels=[3, 3],
                               filters_num=[64, 64],
                               res_in=layer2,
                               layer_idx=3,
                               double_first_stride=True,
                               use_res_conv=True,
                               res_filter=64,
                               res_size=3,
                               res_stride=2)
        t_conv1 = Conv2DTranspose(3, kernel_size=3, strides=2, padding='same', name='t_conv1')(layer3)
        t_conv2 = Conv2DTranspose(3, kernel_size=3, strides=2, padding='same', name='t_conv2')(t_conv1)
        self.model = Model(inputs=visible, outputs=t_conv2)
        self.model.compile(Adam(), loss=MeanSquaredError(), metrics=['accuracy'])

    def fit(self,
            trainX: np.ndarray,
            trainY: np.ndarray,
            batch_size: int,
            epochs: int,
            validation_data: Tuple[np.ndarray, np.ndarray]):
        return self.model.fit(trainX, trainY,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=validation_data)

    def evaluate(self,
                 testX: np.ndarray,
                 testY: np.ndarray,
                 batch_size: int):
        self.model.evaluate(testX, testY, batch_size=batch_size)

    def predict(self, X: np.ndarray):
        return self.model.predict(X)

    def summary(self):
        self.model.summary()

    def plot_model(self, path):
        plot_model(self.model, to_file=path, show_shapes=True)
