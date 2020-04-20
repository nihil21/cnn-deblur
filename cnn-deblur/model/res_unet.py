from tensorflow.keras.models import Model
from model.conv_net import ConvNet, ResUDown, ResUUp, ResUOut
from tensorflow.keras.layers import Input
from typing import Tuple


class ResUNet(ConvNet):

    def __init__(self, input_shape: Tuple[int, int, int]):
        super().__init__()

        # ENCODER
        visible = Input(shape=input_shape)   # 32x32x3

        conv1 = ResUDown(kernels=[3, 3],
                         filters_num=[64, 64],
                         strides=[1, 1],
                         in_layer=visible,
                         layer_idx=1,
                         is_initial=True)  # 32x32x64

        conv2 = ResUDown(kernels=[3, 3],
                         filters_num=[128, 128],
                         strides=[2, 1],
                         in_layer=conv1,
                         layer_idx=2)  # 16x16x128

        conv3 = ResUDown(kernels=[3, 3],
                         filters_num=[256, 256],
                         strides=[2, 1],
                         in_layer=conv2,
                         layer_idx=3)  # 8x8x256

        # BOTTLENECK
        conv4 = ResUDown(kernels=[3, 3],
                         filters_num=[512, 512],
                         strides=[2, 1],
                         in_layer=conv3,
                         layer_idx=5)  # 4x4x512

        # DECODER
        conv5 = ResUUp(kernels=[3, 3],
                       filters_num=[256, 256],
                       strides=[1, 1],
                       in_layer=conv4,
                       concat_layer=conv3,
                       layer_idx=6)  # 8x8x256

        conv6 = ResUUp(kernels=[3, 3],
                       filters_num=[128, 128],
                       strides=[1, 1],
                       in_layer=conv5,
                       concat_layer=conv2,
                       layer_idx=7)  # 16x16x128

        conv7 = ResUUp(kernels=[3, 3],
                       filters_num=[64, 64],
                       strides=[1, 1],
                       in_layer=conv6,
                       concat_layer=conv1,
                       layer_idx=8)  # 32x32x64

        output = ResUOut(conv7)

        self.model = Model(inputs=visible, outputs=output)
