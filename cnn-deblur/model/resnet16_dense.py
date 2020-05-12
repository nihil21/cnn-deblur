from model.conv_net import ConvNet, ResConv, ResConvTranspose
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout, Reshape
from tensorflow.keras.models import Model
from typing import Tuple


class ResNet16Dense(ConvNet):

    def __init__(self, input_shape: Tuple[int, int, int]):
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
