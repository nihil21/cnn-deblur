from tensorflow.keras.models import Model
from model.conv_net import ConvNet, UConvDown, UConvUp
from tensorflow.keras.layers import Input, ZeroPadding2D, Cropping2D
from typing import Tuple


class UNetREDS(ConvNet):

    def __init__(self, input_shape: Tuple[int, int, int]):
        super().__init__()

        # ENCODER
        visible = Input(shape=input_shape)   # 1280x720x3

        # adding 8 rows of pixels (on top and bottom)
        padded = ZeroPadding2D(padding=((0, 0), (8, 8)))(visible)   # 1280x736x3

        conv1 = UConvDown(kernels=[3, 3],
                          filters_num=[16, 16],
                          in_layer=padded,
                          layer_idx=1,
                          middle=False)  # 1280x736x16

        conv2 = UConvDown(kernels=[3, 3],
                          filters_num=[32, 32],
                          in_layer=conv1,
                          layer_idx=2)  # 640x368x32

        conv3 = UConvDown(kernels=[3, 3],
                          filters_num=[64, 64],
                          in_layer=conv2,
                          layer_idx=3)  # 320x184x64

        conv4 = UConvDown(kernels=[3, 3],
                          filters_num=[128, 128],
                          in_layer=conv3,
                          layer_idx=4)  # 160x92x128

        conv5 = UConvDown(kernels=[3, 3],
                          filters_num=[256, 256],
                          in_layer=conv4,
                          layer_idx=5)  # 80x46x256

        # BOTTLENECK
        conv6 = UConvDown(kernels=[3, 3],
                          filters_num=[512, 512],
                          in_layer=conv5,
                          layer_idx=6)  # 40x23x512

        # DECODER
        conv7 = UConvUp(kernels=[3, 3],
                        filters_num=[256, 256],
                        in_layer=conv6,
                        concat_layer=conv5,
                        layer_idx=7)  # 80x46x256

        conv8 = UConvUp(kernels=[3, 3],
                        filters_num=[128, 128],
                        in_layer=conv7,
                        concat_layer=conv4,
                        layer_idx=8)  # 160x92x128

        conv9 = UConvUp(kernels=[3, 3],
                        filters_num=[64, 64],
                        in_layer=conv8,
                        concat_layer=conv3,
                        layer_idx=9)  # 320x184x64

        conv10 = UConvUp(kernels=[3, 3],
                         filters_num=[32, 32],
                         in_layer=conv9,
                         concat_layer=conv2,
                         layer_idx=10)  # 640x368x32

        conv11 = UConvUp(kernels=[3, 3, 1],
                         filters_num=[16, 16, 3],
                         in_layer=conv10,
                         concat_layer=conv1,
                         layer_idx=11)  # 1280x736x3

        cropped = Cropping2D(cropping=((0, 0), (8, 8)))(conv11)

        self.model = Model(inputs=visible, outputs=cropped)

