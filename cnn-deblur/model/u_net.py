from model.conv_net import ConvNet, UConvDown, UConvUp
from tensorflow.keras.layers import Input
from typing import Tuple


class UNet(ConvNet):

    def __init__(self, input_shape: Tuple[int, int, int]):
        super().__init__()
        # ENCODER
        visible = Input(shape=input_shape)
        conv1 = UConvDown(kernels=[3, 3],
                          filters_num=[16, 16],
                          in_layer=visible,
                          layer_idx=1,
                          middle=False)
        conv2 = UConvDown(kernels=[3, 3],
                          filters_num=[32, 32],
                          in_layer=conv1,
                          layer_idx=2)
        conv3 = UConvDown(kernels=[3, 3],
                          filters_num=[64, 64],
                          in_layer=conv2,
                          layer_idx=3)
        conv4 = UConvDown(kernels=[3, 3],
                          filters_num=[128, 128],
                          in_layer=conv3,
                          layer_idx=4)
        # DECODER
        conv5 = UConvUp(kernels=[3, 3],
                        filters_num=[64, 64],
                        in_layer=conv4,
                        concat_layer=conv3,
                        layer_idx=4)
