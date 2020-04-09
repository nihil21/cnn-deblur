from tensorflow.keras.layers import (Layer, Conv2D, Conv2DTranspose, Activation, Add, MaxPooling2D, concatenate)
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import plot_model
from typing import List, Optional


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


def UConvDown(kernels: List[int],
              filters_num: List[int],
              in_layer: Layer,
              layer_idx: int,
              middle: Optional[bool] = True):
    # If the block is not at the input of the network, apply max pooling
    if middle:
        x = MaxPooling2D(pool_size=2, strides=2, name='pool{0:d}'.format(layer_idx))(in_layer)
    else:
        x = in_layer
    n = 0
    for kernel, fltr in zip(kernels, filters_num):
        x = Conv2D(fltr,
                   kernel_size=kernel,
                   activation='relu',
                   padding='same',
                   name='conv{0:d}_{1:d}'.format(layer_idx, n))(x)
        n += 1
    return x


def UConvUp(kernels: List[int],
            filters_num: List[int],
            in_layer: Layer,
            concat_layer: Layer,
            layer_idx: int):
    x = Conv2DTranspose(filters_num[0],
                        kernel_size=2,
                        strides=2,
                        activation='relu',
                        padding='same',
                        name='upsamp{0:d}'.format(layer_idx))(in_layer)

    # concatenation
    x = concatenate([concat_layer, x])

    n = 0
    for kernel, fltr in zip(kernels, filters_num):
        x = Conv2D(fltr,
                   kernel_size=kernel,
                   activation='relu',
                   padding='same',
                   name='conv{0:d}_{1:d}'.format(layer_idx, n))(x)
        n += 1
    return x


class ConvNet:
    """Abstract class representing a generic Convolutional Neural Network"""

    def __init__(self):
        self.model = None

    def fit(self,
            x,
            y,
            batch_size: int,
            epochs: int,
            steps_per_epoch: Optional[int] = None,
            validation_split: Optional[float] = 0.0,
            initial_epoch: Optional[int] = 0,
            callbacks: Optional[List[Callback]] = None):
        return self.model.fit(x, y,
                              batch_size=batch_size,
                              epochs=epochs,
                              steps_per_epoch=steps_per_epoch,
                              validation_split=validation_split,
                              initial_epoch=initial_epoch,
                              callbacks=callbacks)

    def fit_generator(self,
                      train_data,
                      epochs: int,
                      steps_per_epoch: Optional[int] = None,
                      validation_split: Optional[float] = 0.0,
                      initial_epoch: Optional[int] = 0,
                      callbacks: Optional[List[Callback]] = None):
        return self.model.fit(train_data,
                              epochs=epochs,
                              steps_per_epoch=steps_per_epoch,
                              validation_split=validation_split,
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
