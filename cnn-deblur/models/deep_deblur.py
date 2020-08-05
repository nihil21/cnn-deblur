from models.conv_net import ConvNet
from utils.custom_metrics import ssim, psnr
import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Conv2D, Conv2DTranspose, Add, ELU, BatchNormalization, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.losses import logcosh
from tensorflow.keras.optimizers import Adam
from typing import Tuple, List, Optional


def res_block(in_layer: Layer,
              layer_id: str,
              filters: Optional[int] = 64,
              kernels: Optional[int] = 5,
              use_batchnorm: Optional[bool] = True):
    # Block 1
    x = Conv2D(filters=filters,
               kernel_size=kernels,
               padding='same',
               name='res_conv{:s}_1'.format(layer_id))(in_layer)
    if use_batchnorm:
        x = BatchNormalization(name='res_bn{:s}_1'.format(layer_id))(x)
    x = ELU(name='res_elu{:s}_1'.format(layer_id))(x)
    # Block 2
    x = Conv2D(filters=filters,
               kernel_size=kernels,
               padding='same',
               name='res_conv{:s}_2'.format(layer_id))(x)
    if use_batchnorm:
        x = BatchNormalization(name='res_bn{:s}_2'.format(layer_id))(x)
    # Skip connection
    x = Add(name='res_add{:s}'.format(layer_id))([x, in_layer])
    x = ELU(name='res_elu{:s}_2'.format(layer_id))(x)
    return x


class DeepDeblur(ConvNet):
    def __init__(self, input_shape: Tuple[int, int, int]):
        super(DeepDeblur, self).__init__()
        # Coarsest branch
        in_layer3 = Input(shape=(input_shape[0] // 4, input_shape[1] // 4, input_shape[2]),
                          name='in_layer3')
        conv3 = Conv2D(filters=64,
                       kernel_size=5,
                       padding='same',
                       name='conv3')(in_layer3)
        x = conv3
        for i in range(19):
            x = res_block(in_layer=x,
                          layer_id='3_{:d}'.format(i))
        out_layer3 = Conv2D(filters=3,
                            kernel_size=5,
                            padding='same',
                            name='out_layer_3')(x)

        # Middle branch
        in_layer2 = Input(shape=(input_shape[0] // 2, input_shape[1] // 2, input_shape[2]),
                          name='in_layer2')
        up_conv2 = Conv2DTranspose(filters=64,
                                   kernel_size=5,
                                   strides=2,
                                   padding='same')(out_layer3)
        concat2 = concatenate([in_layer2, up_conv2])
        conv2 = Conv2D(filters=64,
                       kernel_size=5,
                       padding='same',
                       name='conv2')(concat2)
        x = conv2
        for i in range(19):
            x = res_block(in_layer=x,
                          layer_id='2_{:d}'.format(i))
        out_layer2 = Conv2D(filters=3,
                            kernel_size=5,
                            padding='same',
                            name='out_layer2')(x)

        # Finest branch
        in_layer1 = Input(shape=input_shape,
                          name='in_layer1')
        up_conv1 = Conv2DTranspose(filters=64,
                                   kernel_size=5,
                                   strides=2,
                                   padding='same')(out_layer2)
        concat1 = concatenate([in_layer1, up_conv1])
        conv1 = Conv2D(filters=64,
                       kernel_size=5,
                       padding='same',
                       name='conv1')(concat1)
        x = conv1
        for i in range(19):
            x = res_block(in_layer=x,
                          layer_id='1_{:d}'.format(i))
        out_layer1 = Conv2D(filters=3,
                            kernel_size=5,
                            padding='same',
                            name='out_layer1')(x)

        # Final model
        self.model = Model(inputs=[in_layer1, in_layer2, in_layer3], outputs=[out_layer1, out_layer2, out_layer3])

        # Define loss function
        K = 3

        def multiscale_logcosh(trueY: List[tf.Tensor], predY: List[tf.Tensor]):
            loss = 0
            for s in range(K):
                scale_shape = trueY[s].shape
                loss += logcosh(trueY[s], predY[s]) / (scale_shape[0] * scale_shape[1] * scale_shape[2])
            return 1./(2. * K) * loss

        self.model.compile(optimizer=Adam(lr=1e-4),
                           loss=multiscale_logcosh,
                           metrics=[ssim, psnr, 'mse', 'mae'])
