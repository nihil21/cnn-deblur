from model.conv_net import ConvNet, ResConv, ResConvTranspose
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from utils.loss_functions import *
from typing import Tuple


class ResNet64(ConvNet):

    def __init__(self, input_shape: Tuple[int, int, int]):
        super().__init__()
        # ENCODER
        visible = Input(shape=input_shape)
        # Convolution layer: Conv + ReLU + BatchNorm
        conv = Conv2D(16,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      activation='relu',
                      name='conv')(visible)
        # conv = BatchNormalization(axis=3, name='bn')(conv)
        # First layer: 2x(Conv + ReLU) + Identity Residual (16 filters)
        layer1 = ResConv(kernels=[3, 3],
                         filters_num=[16, 16],
                         res_in=conv,
                         layer_idx=1)
        # Second layer: 2x(Conv + ReLU) with double first stride + Conv Residual (32 filters)
        layer2 = ResConv(kernels=[3, 3],
                         filters_num=[32, 32],
                         res_in=layer1,
                         layer_idx=2,
                         double_first_stride=True,
                         use_res_conv=True,
                         res_filter=32,
                         res_size=3,
                         res_stride=2)
        # Third layer: same as second layer, but with 64 filters
        layer3 = ResConv(kernels=[3, 3],
                         filters_num=[64, 64],
                         res_in=layer2,
                         layer_idx=3,
                         double_first_stride=True,
                         use_res_conv=True,
                         res_filter=64,
                         res_size=3,
                         res_stride=2)
        # DECODER
        # Forth layer: 2x(DeConv + ReLU) with double last stride + Conv Residual (64 filters)
        layer4 = ResConvTranspose(kernels=[3, 3],
                                  filters_num=[64, 32],
                                  res_in=layer3,
                                  layer_idx=4,
                                  double_last_stride=True,
                                  use_res_tconv=True,
                                  res_filter=32,
                                  res_size=3,
                                  res_stride=2)
        # Fifth layer: same as forth layer, but with 32 filters
        layer5 = ResConvTranspose(kernels=[3, 3],
                                  filters_num=[32, 16],
                                  res_in=layer4,
                                  layer_idx=5,
                                  double_last_stride=True,
                                  use_res_tconv=True,
                                  res_filter=16,
                                  res_size=3,
                                  res_stride=2)
        # Sixth layer: 2x(DeConv + ReLU) + Identity Residual (16 filters)
        layer6 = ResConvTranspose(kernels=[3, 3],
                                  filters_num=[16, 16],
                                  res_in=layer5,
                                  layer_idx=6)
        # Convolution layer: Conv + ReLU
        tconv = Conv2DTranspose(3, kernel_size=3, padding='same', activation='relu')(layer6)

        self.model = Model(inputs=visible, outputs=tconv)
        self.model.compile(Adam(learning_rate=1e-4), loss=ssim_loss, metrics=['accuracy'])
