from tensorflow.keras.models import Model
from model.conv_net import ConvNet
from tensorflow.keras.layers import Input, Layer, Conv2D, Conv2DTranspose, BatchNormalization, Activation, Add
from typing import Tuple, List, Optional


def ResSkipUDown(kernels: List[int],
                 filters_num: List[int],
                 strides: List[int],
                 in_layer: Layer,
                 layer_idx: int,
                 is_initial: Optional[bool] = False):
    x = in_layer

    n = 0
    for krnl, fltr, strd in zip(kernels, filters_num, strides):
        # Update the suffix of layer's name
        layer_suffix = '{0:d}_{1:d}'.format(layer_idx, n)

        # If the block is the initial one, skip batch normalization and ReLU
        if not (is_initial and n == 0):
            x = BatchNormalization(name='bn{0:s}'.format(layer_suffix))(x)
            x = Activation('relu', name='relu{0:s}'.format(layer_suffix))(x)
        x = Conv2D(fltr,
                   kernel_size=krnl,
                   padding='same',
                   strides=strd,
                   name='conv{0:d}_{1:d}'.format(layer_idx, n))(x)
        n += 1

    return x


def ResSkipUUp(kernels: List[int],
               filters_num: List[int],
               strides: List[int],
               in_layer: Layer,
               layer_idx: int,
               res_layer: Optional[bool] = None):
    if res_layer is not None:
        # Residual connection
        x = Add()([in_layer, res_layer])
    else:
        x = in_layer

    n = 0
    for krnl, fltr, strd in zip(kernels, filters_num, strides):
        # Update the suffix of layer's name
        layer_suffix = '{0:d}_{1:d}'.format(layer_idx, n)

        x = BatchNormalization(name='bn{0:s}'.format(layer_suffix))(x)
        x = Activation('relu', name='relu{0:s}'.format(layer_suffix))(x)
        x = Conv2DTranspose(fltr,
                            kernel_size=krnl,
                            padding='same',
                            strides=strd,
                            name='conv{0:d}_{1:d}'.format(layer_idx, n))(x)
        n += 1

    return x


class ResSkipUNet(ConvNet):

    def __init__(self, input_shape: Tuple[int, int, int]):
        super().__init__()

        # ENCODER
        visible = Input(shape=input_shape)  # 32x32x3

        conv1 = ResSkipUDown(kernels=[3, 3],
                             filters_num=[64, 64],
                             strides=[1, 1],
                             in_layer=visible,
                             layer_idx=1,
                             is_initial=True)  # 32x32x64

        conv2 = ResSkipUDown(kernels=[3, 3],
                             filters_num=[128, 128],
                             strides=[2, 1],
                             in_layer=conv1,
                             layer_idx=2)  # 16x16x128

        conv3 = ResSkipUDown(kernels=[3, 3],
                             filters_num=[256, 256],
                             strides=[2, 1],
                             in_layer=conv2,
                             layer_idx=3)  # 8x8x256

        # DOUBLE BOTTLENECK
        conv4 = ResSkipUDown(kernels=[3, 3],
                             filters_num=[512, 512],
                             strides=[2, 1],
                             in_layer=conv3,
                             layer_idx=4)  # 4x4x512

        conv5 = ResSkipUUp(kernels=[3, 3],
                           filters_num=[256, 256],
                           strides=[2, 1],
                           in_layer=conv4,
                           res_layer=None,
                           layer_idx=5)  # 8x8x256

        # DECODER
        conv6 = ResSkipUUp(kernels=[3, 3],
                           filters_num=[128, 128],
                           strides=[2, 1],
                           in_layer=conv5,
                           res_layer=conv3,
                           layer_idx=6)  # 16x16x128

        conv7 = ResSkipUUp(kernels=[3, 3],
                           filters_num=[64, 64],
                           strides=[2, 1],
                           in_layer=conv6,
                           res_layer=conv2,
                           layer_idx=7)  # 32x32x64

        conv8 = ResSkipUUp(kernels=[3, 3],
                           filters_num=[3, 3],
                           strides=[1, 1],
                           in_layer=conv7,
                           res_layer=conv1,
                           layer_idx=8)  # 32x32x3

        output = conv8

        self.model = Model(inputs=visible, outputs=output)

