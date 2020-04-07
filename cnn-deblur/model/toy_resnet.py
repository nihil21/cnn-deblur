from model.conv_net import ConvNet, ResConv
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from typing import Tuple


class ToyResNet(ConvNet):

    def __init__(self, input_shape: Tuple[int, int, int]):
        super().__init__()
        visible = Input(shape=input_shape)
        # Convolution layer: Conv + BatchNorm + ReLU
        conv = Conv2D(16,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      name='conv')(visible)
        conv = BatchNormalization(axis=3, name='bn')(conv)
        conv = Activation('relu', name='relu')(conv)
        # First layer: 2x(Conv + BatchNorm) + Identity Residual (16 filters)
        layer1 = ResConv(kernels=[3, 3],
                         filters_num=[16, 16],
                         res_in=conv,
                         layer_idx=1)
        # Second layer: 2x(Conv + BatchNorm) which double first stride + Conv Residual (32 filters)
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
        t_conv1 = Conv2DTranspose(3, kernel_size=3, strides=2, padding='same', name='t_conv1')(layer3)
        t_conv2 = Conv2DTranspose(3, kernel_size=3, strides=2, padding='same', name='t_conv2')(t_conv1)
        self.model = Model(inputs=visible, outputs=t_conv2)
        self.model.compile(Adam(), loss=MeanSquaredError(), metrics=['accuracy'])
