from tensorflow.keras.models import Model
from model.conv_net import ConvNet, UConvDown, UConvUp
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from utils.loss_functions import *
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError, KLDivergence
from tensorflow.keras.metrics import binary_accuracy
from typing import Tuple, Optional


class UNet(ConvNet):

    def __init__(self, input_shape: Tuple[int, int, int], loss: Optional[str] = 'content_loss'):
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
                        layer_idx=5)

        conv6 = UConvUp(kernels=[3, 3],
                        filters_num=[32, 32],
                        in_layer=conv5,
                        concat_layer=conv2,
                        layer_idx=6)

        conv7 = UConvUp(kernels=[3, 3],
                        filters_num=[16, 16],
                        in_layer=conv6,
                        concat_layer=conv1,
                        layer_idx=7)

        output = Conv2D(kernel_size=1,
                        filters=3,
                        activation='relu',
                        padding='same',
                        name='output'.format(8))(conv7)

        self.model = Model(inputs=visible, outputs=output)

        loss_dict = dict({
            'ssim_loss': ssim_loss,
            'content_loss': content_loss,
            'mix_loss': mix_loss,
            'psnr_loss': psnr_loss,
            'cross_entropy': BinaryCrossentropy(),
            'mse': MeanSquaredError(),
            'kld': KLDivergence()
        })
        self.model.compile(Adam(learning_rate=1e-4), loss=loss_dict[loss], metrics=[binary_accuracy])
