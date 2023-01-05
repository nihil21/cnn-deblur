import typing

from tensorflow import keras

from .conv_net import ConvNet


def ResUDown(
        kernels: typing.List[int],
        filters_num: typing.List[int],
        strides: typing.List[int],
        in_layer: keras.layers.Layer,
        layer_idx: int,
        is_initial: bool = False
):
    x = in_layer

    n = 0
    for krnl, fltr, strd in zip(kernels, filters_num, strides):
        # Update the suffix of layer's name
        layer_suffix = '{0:d}_{1:d}'.format(layer_idx, n)

        # If the block is the initial one, skip batch normalization and ReLU
        if not (is_initial and n == 0):
            x = keras.layers.BatchNormalization(name='bn{0:s}'.format(layer_suffix))(x)
            x = keras.layers.Activation('relu', name='relu{0:s}'.format(layer_suffix))(x)
        x = keras.layers.Conv2D(
            fltr,
            kernel_size=krnl,
            padding='same',
            strides=strd,
            name='conv{0:d}_{1:d}'.format(layer_idx, n)
        )(x)
        n += 1

    # Residual connection
    res_layer = keras.layers.Conv2D(
        filters_num[0],
        kernel_size=1,
        padding='same',
        strides=strides[0],
        name='res_conv{0:d}'.format(layer_idx)
    )(in_layer)
    x = keras.layers.Add()([x, res_layer])

    return x


def ResUUp(
        kernels: typing.List[int],
        filters_num: typing.List[int],
        strides: typing.List[int],
        in_layer: keras.layers.Layer,
        concat_layer: keras.layers.Layer,
        layer_idx: int
):
    # Upsampling by transposed convolution
    x = keras.layers.Conv2DTranspose(
        filters_num[0],
        kernel_size=3,
        strides=2,
        activation='relu',
        padding='same',
        name='upsamp{0:d}'.format(layer_idx)
    )(in_layer)
    # Concatenation
    x = keras.layers.concatenate([concat_layer, x])
    # Residual layer
    res_layer = keras.layers.Conv2DTranspose(
        filters_num[0],
        kernel_size=1,
        strides=1,
        activation='relu',
        padding='same',
        name='res_upsamp{0:d}'.format(layer_idx)
    )(x)

    n = 0
    for krnl, fltr, strd in zip(kernels, filters_num, strides):
        # Update the suffix of layer's name
        layer_suffix = '{0:d}_{1:d}'.format(layer_idx, n)

        x = keras.layers.BatchNormalization(name='bn{0:s}'.format(layer_suffix))(x)
        x = keras.layers.Activation('relu', name='relu{0:s}'.format(layer_suffix))(x)
        x = keras.layers.Conv2D(
            fltr,
            kernel_size=krnl,
            padding='same',
            strides=strd,
            name='conv{0:d}_{1:d}'.format(layer_idx, n)
        )(x)
        n += 1

    # Residual connection
    x = keras.layers.Add()([x, res_layer])

    return x


def ResUOut(in_layer: keras.layers.Layer):
    x = keras.layers.Conv2D(3, kernel_size=1, strides=1, padding='same', name='conv_out')(in_layer)
    return keras.layers.Activation('sigmoid', name='sigmoid')(x)


class ResUNet16(ConvNet):

    def __init__(self, input_shape: typing.Tuple[int, int, int]):
        super().__init__()

        # ENCODER
        visible = keras.layers.Input(shape=input_shape)   # 512x288x3

        conv1 = ResUDown(kernels=[3, 3],
                         filters_num=[64, 64],
                         strides=[1, 1],
                         in_layer=visible,
                         layer_idx=1,
                         is_initial=True)  # 512x288x64

        conv2 = ResUDown(kernels=[3, 3],
                         filters_num=[128, 128],
                         strides=[2, 1],
                         in_layer=conv1,
                         layer_idx=2)  # 256x144x128

        conv3 = ResUDown(kernels=[3, 3],
                         filters_num=[256, 256],
                         strides=[2, 1],
                         in_layer=conv2,
                         layer_idx=3)  # 128x72x256

        conv4 = ResUDown(kernels=[3, 3],
                         filters_num=[512, 512],
                         strides=[2, 1],
                         in_layer=conv3,
                         layer_idx=4)  # 64x36x512

        # DECODER
        conv5 = ResUUp(kernels=[3, 3],
                       filters_num=[256, 256],
                       strides=[1, 1],
                       in_layer=conv4,
                       concat_layer=conv3,
                       layer_idx=6)  # 128x72x256

        conv6 = ResUUp(kernels=[3, 3],
                       filters_num=[128, 128],
                       strides=[1, 1],
                       in_layer=conv5,
                       concat_layer=conv2,
                       layer_idx=7)  # 256x144x128

        conv7 = ResUUp(kernels=[3, 3],
                       filters_num=[64, 64],
                       strides=[1, 1],
                       in_layer=conv6,
                       concat_layer=conv1,
                       layer_idx=8)  # 512x288x64

        output = ResUOut(conv7)

        self.model = keras.models.Model(inputs=visible, outputs=output)
