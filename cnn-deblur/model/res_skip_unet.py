from tensorflow.keras.models import Model
from model.conv_net import ConvNet, ResSkipUDown, ResSkipUUp
from tensorflow.keras.layers import Input
from typing import Tuple


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
                           filters_num=[64, 64],
                           strides=[1, 1],
                           in_layer=conv7,
                           res_layer=conv1,
                           layer_idx=8)  # 32x32x3

        output = conv8

        self.model = Model(inputs=visible, outputs=output)


conv_net = ResSkipUNet(input_shape=(32, 32, 3))
conv_net.plot_model("/tmp/model.png")
