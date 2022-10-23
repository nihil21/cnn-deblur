from tensorflow.keras.models import Model
from models.conv_net import ConvNet
from tensorflow.keras.layers import Input, Layer, Conv2D, BatchNormalization, Activation, Subtract, concatenate
import typing


def ConvBRNRelu(kernel: int,
                filter_num: int,
                stride: int,
                in_layer: Layer,
                layer_idx: str,
                blocks_number: int,
                dilation_rate: int = 1):

    x = in_layer

    for i in range(1, blocks_number + 1):
        # Update the suffix of layer's name
        layer_suffix = '{0:s}_{1:d}'.format(layer_idx, i)

        x = Conv2D(filter_num,
                   kernel_size=kernel,
                   padding='same',
                   strides=stride,
                   dilation_rate=dilation_rate,
                   name='conv{0:s}_{1:d}'.format(layer_idx, i))(x)
        # TODO BRN
        x = BatchNormalization(name='bn{0:s}'.format(layer_suffix))(x)
        x = Activation('relu', name='relu{0:s}'.format(layer_suffix))(x)

    return x


class BRDNet(ConvNet):

    def __init__(self, input_shape: typing.Tuple[int, int, int]):
        super().__init__()

        self.visible = Input(shape=input_shape)  # 240x320x3

        # UPPER

        upper1 = ConvBRNRelu(kernel=3,
                             filter_num=64,
                             stride=1,
                             in_layer=self.visible,
                             layer_idx="up_1",
                             blocks_number=16)

        upper2 = Conv2D(kernel_size=3,
                        filters=3,
                        padding='same',
                        strides=1,
                        name='up_2')(upper1)

        upper3 = Subtract(name="up_3")([self.visible, upper2])

        # LOWER

        lower1 = ConvBRNRelu(kernel=3,
                             filter_num=64,
                             stride=1,
                             in_layer=self.visible,
                             layer_idx='low_1',
                             blocks_number=1)

        lower2 = ConvBRNRelu(kernel=3,
                             filter_num=64,
                             stride=1,
                             in_layer=lower1,
                             layer_idx='low_2',
                             blocks_number=7,
                             dilation_rate=2)

        lower3 = ConvBRNRelu(kernel=3,
                             filter_num=64,
                             stride=1,
                             in_layer=lower2,
                             layer_idx='low_3',
                             blocks_number=1)

        lower4 = ConvBRNRelu(kernel=3,
                             filter_num=64,
                             stride=1,
                             in_layer=lower3,
                             layer_idx='low_4',
                             blocks_number=7,
                             dilation_rate=2)

        lower5 = Conv2D(kernel_size=3,
                        filters=3,
                        padding='same',
                        strides=1,
                        name='low_5')(lower4)

        lower6 = Subtract(name="low_6")([self.visible, lower5])

        # UNION
        union1 = concatenate([upper3, lower6], name="union_1")

        union2 = Conv2D(kernel_size=3,
                        filters=3,
                        padding='same',
                        strides=1,
                        name='union_2')(union1)

        output = Subtract(name="output")([self.visible, union2])

        self.model = Model(inputs=self.visible, outputs=output)

    """def my_compile(self, lr: Optional[float] = 1e-4):

        metric_list = [ssim_metric,
                       psnr_metric,
                       'mse',
                       'mae',
                       'accuracy']

        def custom_loss_wrapper(visible):
            def custom_loss(trueY, predY):
                return mse(predY - visible, visible - trueY)
            return custom_loss

        self.model.compile(Adam(learning_rate=lr),
                           loss=custom_loss_wrapper(self.visible),
                           metrics=metric_list)"""
