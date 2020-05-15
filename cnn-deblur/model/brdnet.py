from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from model.conv_net import ConvNet, ConvBRNRelu, Add
from tensorflow.keras.layers import Input, Conv2D, concatenate
from typing import Tuple, Optional

from model.custom_losses_metrics import ssim_metric, psnr_metric


class BRDNet(ConvNet):

    def __init__(self, input_shape: Tuple[int, int, int]):
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

        upper3 = Add(name="up_3")([self.visible, upper2])

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

        lower6 = Add(name="low_6")([self.visible, lower5])

        # UNION

        union1 = concatenate([upper3, lower6], name="union_1")

        union2 = Conv2D(kernel_size=3,
                        filters=3,
                        padding='same',
                        strides=1,
                        name='union_2')(union1)

        output = Add(name="output")([self.visible, union2])

        self.model = Model(inputs=self.visible, outputs=output)

    def my_compile(self, lr: Optional[float] = 1e-4):

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
                           metrics=metric_list)
