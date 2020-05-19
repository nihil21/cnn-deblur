from model.conv_net import ConvNet
from tensorflow.keras.layers import Input, Layer, Conv2D, Conv2DTranspose, Add, ELU, BatchNormalization
from tensorflow.keras.models import Model
# from tensorflow.keras.constraints import min_max_norm
from typing import Tuple, List, Optional


def encode(in_layer: Layer, num_layers: Optional[int] = 15, num_filters: Optional[int] = 64) -> List[Layer]:
    layers: List[Layer] = []
    x = in_layer
    for i in range(num_layers):
        """if i == 0:
            stride = 2
        else:
            stride = 1"""
        x = Conv2D(filters=num_filters,
                   kernel_size=3,
                   strides=1,
                   padding='same',
                   # kernel_constraint=min_max_norm(min_value=0., max_value=1.),
                   name=f'encode_conv{i}')(x)
        x = ELU(name=f'encode_elu{i}')(x)
        x = BatchNormalization(name=f'encode_bn{i}')(x)
        layers.append(x)
    return layers


def decode(res_layers: List[Layer], num_layers: Optional[int] = 15, num_filters: Optional[int] = 64) -> List[Layer]:
    layers: List[Layer] = []
    res_layers.reverse()
    x = res_layers[0]
    for i in range(num_layers):
        x = Conv2DTranspose(filters=num_filters,
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            # kernel_constraint=min_max_norm(min_value=0., max_value=1.),
                            name=f'decode_conv{i}')(x)
        if i % 2 != 0:
            x = Add(name=f'decode_skip{i}')([x, res_layers[i]])
        x = ELU(name=f'decode_elu{i}')(x)
        x = BatchNormalization(name=f'decode_bn{i}')(x)
        layers.append(x)

    return layers


class REDNet30(ConvNet):
    def __init__(self, input_shape: Tuple[int, int, int]):
        super().__init__()
        # ENCODER
        visible = Input(input_shape)
        encode_layers = encode(visible)
        # DECODER
        decode_layers = decode(encode_layers)
        output = Conv2DTranspose(filters=3,
                                 kernel_size=1,
                                 strides=1,
                                 padding='same',
                                 # kernel_constraint=min_max_norm(min_value=0., max_value=1.),
                                 name='output')(decode_layers[-1])
        output = Add()([output, visible])
        # output = ELU()(output)

        self.model = Model(inputs=visible, outputs=output)
