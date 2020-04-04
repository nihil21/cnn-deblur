from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Layer, Input, Conv2D, Conv2DTranspose, Activation, Add,
                                     AveragePooling2D, Flatten, Dense, Reshape, UpSampling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.utils import plot_model
import numpy as np
from typing import List, Tuple, Optional


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
        res_in = Conv2D(res_filter, kernel_size=res_size, strides=res_stride, padding='same')(res_in)

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
            """
            x = BatchNormalization(axis=3, name='bn{0:s}'.format(layer_suffix))(x)
            x = Activation('relu', name='relu{0:s}'.format(layer_suffix))(x)
            """

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
        res_in = Conv2DTranspose(res_filter, kernel_size=res_size, strides=res_stride, padding='same')(res_in)

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
            """
            x = BatchNormalization(axis=3, name='bn{0:s}'.format(layer_suffix))(x)
            x = Activation('relu', name='relu{0:s}'.format(layer_suffix))(x)
            """

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
        # Convolution layer: Conv + BatchNorm + ReLU
        conv = Conv2D(16,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      name='conv',
                      activation='relu')(visible)
        """
        conv = BatchNormalization(axis=3, name='bn')(conv)
        conv = Activation('relu', name='relu')(conv)
        """
        # First layer: 2x(Conv + BatchNorm) + Identity Residual (16 filters)
        layer1 = ResConv(kernels=[3, 3],
                         filters_num=[16, 16],
                         res_in=conv,
                         layer_idx=1)
        # Second layer: 2x(Conv + BatchNorm) which double first stride + Conv Residual (32 filters)
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
        # Average pooling + flatten
        avg_pool = AveragePooling2D(pool_size=(8, 8))(layer3)
        flat = Flatten()(avg_pool)
        # Dense bottleneck
        dense = Dense(64, input_shape=(64,), activation='relu')(flat)
        # DECODER
        reshape = Reshape((1, 1, 64))(dense)
        upsample = UpSampling2D(8, interpolation='nearest')(reshape)
        layer4 = ResConvTranspose(kernels=[3, 3],
                                  filters_num=[64, 32],
                                  res_in=upsample,
                                  layer_idx=4,
                                  double_last_stride=True,
                                  use_res_tconv=True,
                                  res_filter=32,
                                  res_size=3,
                                  res_stride=2)
        layer5 = ResConvTranspose(kernels=[3, 3],
                                  filters_num=[32, 16],
                                  res_in=layer4,
                                  layer_idx=5,
                                  double_last_stride=True,
                                  use_res_tconv=True,
                                  res_filter=16,
                                  res_size=3,
                                  res_stride=2)
        layer6 = ResConvTranspose(kernels=[3, 3],
                                  filters_num=[16, 16],
                                  res_in=layer5,
                                  layer_idx=6)
        tconv = Conv2DTranspose(3, kernel_size=3, padding='same', activation='relu')(layer6)

        self.model = Model(inputs=visible, outputs=tconv)
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
