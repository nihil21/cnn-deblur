import typing

from tensorflow import keras

from .conv_net import ConvNet


def UConvDown(
        kernels: typing.List[int],
        filters_num: typing.List[int],
        in_layer: keras.layers.Layer,
        layer_idx: int,
        middle: bool = True
):
    # If the block is not at the input of the network, apply max pooling
    if middle:
        x = keras.layers.MaxPooling2D(pool_size=2, strides=2, name='pool{0:d}'.format(layer_idx))(in_layer)
    else:
        x = in_layer
    n = 0
    for kernel, fltr in zip(kernels, filters_num):
        x = keras.layers.Conv2D(
            fltr,
            kernel_size=kernel,
            activation='relu',
            padding='same',
            name='conv{0:d}_{1:d}'.format(layer_idx, n)
        )(x)
        n += 1
    return x


def UConvUp(
        kernels: typing.List[int],
        filters_num: typing.List[int],
        in_layer: keras.layers.Layer,
        concat_layer: keras.layers.Layer,
        layer_idx: int
):
    # Upsampling by transposed convolution
    x = keras.layers.Conv2DTranspose(
        filters_num[0],
        kernel_size=2,
        strides=2,
        activation='relu',
        padding='same',
        name='upsamp{0:d}'.format(layer_idx)
    )(in_layer)
    # Concatenation
    x = keras.layers.concatenate([concat_layer, x])

    n = 0
    for kernel, fltr in zip(kernels, filters_num):
        x = keras.layers.Conv2D(
            fltr,
            kernel_size=kernel,
            activation='relu',
            padding='same',
            name='conv{0:d}_{1:d}'.format(layer_idx, n)
        )(x)
        n += 1
    return x


class UNet16(ConvNet):

    def __init__(self, input_shape: typing.Tuple[int, int, int]):
        super().__init__()

        # ENCODER
        visible = keras.layers.Input(shape=input_shape)

        conv1 = UConvDown(
            kernels=[3, 3],
            filters_num=[16, 16],
            in_layer=visible,
            layer_idx=1,
            middle=False
        )

        conv2 = UConvDown(
            kernels=[3, 3],
            filters_num=[32, 32],
            in_layer=conv1,
            layer_idx=2
        )

        conv3 = UConvDown(
            kernels=[3, 3],
            filters_num=[64, 64],
            in_layer=conv2,
            layer_idx=3
        )

        # BOTTLENECK
        conv4 = UConvDown(
            kernels=[3, 3],
            filters_num=[128, 128],
            in_layer=conv3,
            layer_idx=4
        )

        # DECODER
        conv5 = UConvUp(
            kernels=[3, 3],
            filters_num=[64, 64],
            in_layer=conv4,
            concat_layer=conv3,
            layer_idx=5
        )

        conv6 = UConvUp(
            kernels=[3, 3],
            filters_num=[32, 32],
            in_layer=conv5,
            concat_layer=conv2,
            layer_idx=6
        )

        # Output layer is comprised in 'conv7' by adding a third kernel size '1' in 'kernels' list, and
        # a third filter number '3' in 'filters_num' list
        conv7 = UConvUp(
            kernels=[3, 3, 1],
            filters_num=[16, 16, 3],
            in_layer=conv6,
            concat_layer=conv1,
            layer_idx=7
        )

        self.model = keras.models.Model(inputs=visible, outputs=conv7)


class UNet20(ConvNet):

    def __init__(self, input_shape: typing.Tuple[int, int, int]):
        super().__init__()

        # ENCODER
        visible = keras.layers.Input(shape=input_shape)   # 512x288x3

        conv1 = UConvDown(
            kernels=[3, 3],
            filters_num=[16, 16],
            in_layer=visible,
            layer_idx=1,
            middle=False
        )  # 512x288x16

        conv2 = UConvDown(
            kernels=[3, 3],
            filters_num=[32, 32],
            in_layer=conv1,
            layer_idx=2
        )  # 256x144x32

        conv3 = UConvDown(
            kernels=[3, 3],
            filters_num=[64, 64],
            in_layer=conv2,
            layer_idx=3
        )  # 128x72x64

        conv4 = UConvDown(
            kernels=[3, 3],
            filters_num=[128, 128],
            in_layer=conv3,
            layer_idx=4
        )  # 64x36x128

        # BOTTLENECK
        conv5 = UConvDown(
            kernels=[3, 3],
            filters_num=[256, 256],
            in_layer=conv4,
            layer_idx=5
        )  # 32x18x256

        # DECODER
        conv6 = UConvUp(
            kernels=[3, 3],
            filters_num=[128, 128],
            in_layer=conv5,
            concat_layer=conv4,
            layer_idx=6
        )  # 64x36x128

        conv7 = UConvUp(
            kernels=[3, 3],
            filters_num=[64, 64],
            in_layer=conv6,
            concat_layer=conv3,
            layer_idx=7
        )  # 128x72x64

        conv8 = UConvUp(
            kernels=[3, 3],
            filters_num=[32, 32],
            in_layer=conv7,
            concat_layer=conv2,
            layer_idx=8
        )  # 256x144x32

        conv9 = UConvUp(
            kernels=[3, 3, 1],
            filters_num=[16, 16, 3],
            in_layer=conv8,
            concat_layer=conv1,
            layer_idx=9
        )  # 512x288x3

        self.model = keras.models.Model(inputs=visible, outputs=conv9)
