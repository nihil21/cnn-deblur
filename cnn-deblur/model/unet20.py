from tensorflow.keras.models import Model
from model.conv_net import ConvNet, UConvDown, UConvUp
from tensorflow.keras.layers import Input
from typing import Tuple


class UNet20(ConvNet):

    def __init__(self, input_shape: Tuple[int, int, int]):
        super().__init__()

        # ENCODER
        visible = Input(shape=input_shape)   # 512x288x3

        conv1 = UConvDown(kernels=[3, 3],
                          filters_num=[16, 16],
                          in_layer=visible,
                          layer_idx=1,
                          middle=False)  # 512x288x16

        conv2 = UConvDown(kernels=[3, 3],
                          filters_num=[32, 32],
                          in_layer=conv1,
                          layer_idx=2)  # 256x144x32

        conv3 = UConvDown(kernels=[3, 3],
                          filters_num=[64, 64],
                          in_layer=conv2,
                          layer_idx=3)  # 128x72x64

        conv4 = UConvDown(kernels=[3, 3],
                          filters_num=[128, 128],
                          in_layer=conv3,
                          layer_idx=4)  # 64x36x128

        # BOTTLENECK
        conv5 = UConvDown(kernels=[3, 3],
                          filters_num=[256, 256],
                          in_layer=conv4,
                          layer_idx=5)  # 32x18x256

        # DECODER
        conv6 = UConvUp(kernels=[3, 3],
                        filters_num=[128, 128],
                        in_layer=conv5,
                        concat_layer=conv4,
                        layer_idx=6)  # 64x36x128

        conv7 = UConvUp(kernels=[3, 3],
                        filters_num=[64, 64],
                        in_layer=conv6,
                        concat_layer=conv3,
                        layer_idx=7)  # 128x72x64

        conv8 = UConvUp(kernels=[3, 3],
                        filters_num=[32, 32],
                        in_layer=conv7,
                        concat_layer=conv2,
                        layer_idx=8)  # 256x144x32

        conv9 = UConvUp(kernels=[3, 3, 1],
                        filters_num=[16, 16, 3],
                        in_layer=conv8,
                        concat_layer=conv1,
                        layer_idx=9)  # 512x288x3

        self.model = Model(inputs=visible, outputs=conv9)
