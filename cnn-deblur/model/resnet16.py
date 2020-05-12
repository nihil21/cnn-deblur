from model.conv_net import ConvNet, ResConv, ResConvTranspose
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from typing import Tuple


class ResNet16(ConvNet):

    def __init__(self, input_shape: Tuple[int, int, int]):
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
