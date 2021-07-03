from models.conv_net import ConvNet
from tensorflow.keras.layers import (Input, AveragePooling2D, Flatten, Dense, Dropout, Reshape, Layer,
                                     Conv2D, Conv2DTranspose, BatchNormalization, Activation, Add)
from tensorflow.keras.models import Model
import typing


def ResConv(kernels: typing.List[int],
            depths: typing.List[int],
            strides: typing.List[int],
            in_layer: Layer,
            layer_idx: int):
    x = in_layer
    n = 0
    for k, d, s in zip(kernels, depths, strides):
        # Update the suffix of layer's name
        layer_suffix = '{0:d}_{1:d}'.format(layer_idx, n)

        x = Conv2D(d,
                   kernel_size=k,
                   strides=s,
                   padding='same',
                   name='conv{0:d}_{1:d}'.format(layer_idx, n))(x)
        x = BatchNormalization(name='bn{0:s}'.format(layer_suffix))(x)
        x = Activation('relu', name='relu{0:s}'.format(layer_suffix))(x)

        n += 1

    # Residual connection
    res_layer = Conv2D(depths[0],
                       kernel_size=1,
                       strides=strides[0],
                       padding='same',
                       name='res_conv{0:d}'.format(layer_idx))(in_layer)
    x = Add()([x, res_layer])

    return x


def ResConvTranspose(kernels: typing.List[int],
                     depths: typing.List[int],
                     strides: typing.List[int],
                     in_layer: Layer,
                     layer_idx: int):
    x = in_layer
    n = 0
    for k, d, s in zip(kernels, depths, strides):
        # Update the suffix of layer's name
        layer_suffix = '{0:d}_{1:d}'.format(layer_idx, n)

        x = Conv2DTranspose(d,
                            kernel_size=k,
                            strides=s,
                            padding='same',
                            name='conv{0:d}_{1:d}'.format(layer_idx, n))(x)
        x = BatchNormalization(name='bn{0:s}'.format(layer_suffix))(x)
        x = Activation('relu', name='relu{0:s}'.format(layer_suffix))(x)

        n += 1

    # Residual connection
    res_layer = Conv2DTranspose(depths[-1],
                                kernel_size=1,
                                strides=strides[-1],
                                padding='same',
                                name='res_conv{0:d}'.format(layer_idx))(in_layer)
    x = Add()([x, res_layer])

    return x


class ResNet16(ConvNet):

    def __init__(self, input_shape: typing.Tuple[int, int, int]):
        super().__init__()
        # ENCODER
        visible = Input(shape=input_shape)
        layer1 = ResConv(kernels=[3, 3],
                         depths=[64, 64],
                         strides=[1, 1],
                         in_layer=visible,
                         layer_idx=1)

        layer2 = ResConv(kernels=[3, 3],
                         depths=[128, 128],
                         strides=[2, 1],
                         in_layer=layer1,
                         layer_idx=2)

        layer3 = ResConv(kernels=[3, 3],
                         depths=[256, 256],
                         strides=[2, 1],
                         in_layer=layer2,
                         layer_idx=3)

        # BOTTLENECK
        layer4 = ResConv(kernels=[3, 3],
                         depths=[512, 512],
                         strides=[2, 1],
                         in_layer=layer3,
                         layer_idx=4)
        
        # DECODER
        layer5 = ResConvTranspose(kernels=[3, 3],
                                  depths=[256, 256],
                                  strides=[1, 2],
                                  in_layer=layer4,
                                  layer_idx=5)

        layer6 = ResConvTranspose(kernels=[3, 3],
                                  depths=[128, 128],
                                  strides=[1, 2],
                                  in_layer=layer5,
                                  layer_idx=6)

        layer7 = ResConvTranspose(kernels=[3, 3],
                                  depths=[64, 64],
                                  strides=[1, 2],
                                  in_layer=layer6,
                                  layer_idx=7)

        layer8 = ResConvTranspose(kernels=[1],
                                  depths=[3],
                                  strides=[1],
                                  in_layer=layer7,
                                  layer_idx=8)

        self.model = Model(inputs=visible, outputs=layer8)


class ResNet16Dense(ConvNet):

    def __init__(self, input_shape: typing.Tuple[int, int, int]):
        super().__init__()
        # ENCODER
        visible = Input(shape=input_shape)
        # ENCODER
        layer1 = ResConv(kernels=[3, 3],
                         depths=[64, 64],
                         strides=[1, 1],
                         in_layer=visible,
                         layer_idx=1)

        layer2 = ResConv(kernels=[3, 3],
                         depths=[128, 128],
                         strides=[2, 1],
                         in_layer=layer1,
                         layer_idx=2)

        layer3 = ResConv(kernels=[3, 3],
                         depths=[256, 256],
                         strides=[2, 1],
                         in_layer=layer2,
                         layer_idx=3)

        # DENSE BOTTLENECK
        avg_pool = AveragePooling2D(pool_size=(8, 8))(layer3)
        flat = Flatten()(avg_pool)
        dense = Dense(256, input_shape=(256,), activation='softmax')(flat)
        drop = Dropout(0.2)(dense)
        reshape = Reshape((16, 16, 1))(drop)
        # DECODER
        layer4 = ResConvTranspose(kernels=[3, 3],
                                  depths=[128, 64],
                                  strides=[1, 2],
                                  in_layer=reshape,
                                  layer_idx=4)

        layer5 = ResConvTranspose(kernels=[3, 3],
                                  depths=[64, 3],
                                  strides=[1, 1],
                                  in_layer=layer4,
                                  layer_idx=5)

        self.model = Model(inputs=visible, outputs=layer5)


class ResNet20(ConvNet):

    def __init__(self, input_shape: typing.Tuple[int, int, int]):
        super().__init__()
        # ENCODER
        visible = Input(shape=input_shape)
        layer1 = ResConv(kernels=[3, 3],
                         depths=[64, 64],
                         strides=[1, 1],
                         in_layer=visible,
                         layer_idx=1)

        layer2 = ResConv(kernels=[3, 3],
                         depths=[128, 128],
                         strides=[2, 1],
                         in_layer=layer1,
                         layer_idx=2)

        layer3 = ResConv(kernels=[3, 3],
                         depths=[256, 256],
                         strides=[2, 1],
                         in_layer=layer2,
                         layer_idx=3)

        layer4 = ResConv(kernels=[3, 3],
                         depths=[512, 512],
                         strides=[2, 1],
                         in_layer=layer3,
                         layer_idx=4)

        # BOTTLENECK
        layer5 = ResConv(kernels=[3, 3],
                         depths=[1024, 1024],
                         strides=[2, 1],
                         in_layer=layer4,
                         layer_idx=5)

        # DECODER
        layer6 = ResConvTranspose(kernels=[3, 3],
                                  depths=[512, 512],
                                  strides=[1, 2],
                                  in_layer=layer5,
                                  layer_idx=6)

        layer7 = ResConvTranspose(kernels=[3, 3],
                                  depths=[256, 256],
                                  strides=[1, 2],
                                  in_layer=layer6,
                                  layer_idx=7)

        layer8 = ResConvTranspose(kernels=[3, 3],
                                  depths=[128, 128],
                                  strides=[1, 2],
                                  in_layer=layer7,
                                  layer_idx=8)

        layer9 = ResConvTranspose(kernels=[3, 3],
                                  depths=[64, 64],
                                  strides=[1, 2],
                                  in_layer=layer8,
                                  layer_idx=9)

        layer10 = ResConvTranspose(kernels=[1],
                                   depths=[3],
                                   strides=[1],
                                   in_layer=layer9,
                                   layer_idx=10)

        self.model = Model(inputs=visible, outputs=layer10)