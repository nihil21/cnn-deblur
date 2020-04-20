from tensorflow.keras.models import Model
from model.conv_net import ConvNet, ResUDown, ResUUp, ResUOut
from tensorflow.keras.layers import Input
from typing import Tuple


class ResUNet20(ConvNet):

    def __init__(self, input_shape: Tuple[int, int, int]):
        super().__init__()

        # ENCODER
        visible = Input(shape=input_shape)   # 512x288x3

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

        # BOTTLENECK
        conv5 = ResUDown(kernels=[3, 3],
                         filters_num=[1024, 1024],
                         strides=[2, 1],
                         in_layer=conv4,
                         layer_idx=5)  # 32x18x1024

        # DECODER
        conv6 = ResUUp(kernels=[3, 3],
                       filters_num=[512, 512],
                       strides=[1, 1],
                       in_layer=conv5,
                       concat_layer=conv4,
                       layer_idx=6)  # 64x36x512

        conv7 = ResUUp(kernels=[3, 3],
                       filters_num=[256, 256],
                       strides=[1, 1],
                       in_layer=conv6,
                       concat_layer=conv3,
                       layer_idx=7)  # 128x72x256

        conv8 = ResUUp(kernels=[3, 3],
                       filters_num=[128, 128],
                       strides=[1, 1],
                       in_layer=conv7,
                       concat_layer=conv2,
                       layer_idx=8)  # 256x144x128

        conv9 = ResUUp(kernels=[3, 3],
                       filters_num=[64, 64],
                       strides=[1, 1],
                       in_layer=conv8,
                       concat_layer=conv1,
                       layer_idx=9)  # 512x288x64

        output = ResUOut(conv9)

        self.model = Model(inputs=visible, outputs=output)
