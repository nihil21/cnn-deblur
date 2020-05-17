from model.conv_net import ConvNet
from tensorflow.keras.layers import Input, Layer, Conv2D, Conv2DTranspose, Add
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import min_max_norm
from typing import Tuple, List, Optional


def encode(in_layer: Layer, num_layers: Optional[int] = 15, num_filters: Optional[int] = 64) -> List[Layer]:
    layers: List[Layer] = [in_layer]
    x = in_layer
    for i in range(num_layers):
        if i == 0:
            stride = 2
        else:
            stride = 1
        x = Conv2D(filters=num_filters,
                   kernel_size=3,
                   strides=stride,
                   padding='same',
                   activation='relu',
                   kernel_constraint=min_max_norm(min_value=0., max_value=1.),
                   name='encode{0:d}'.format(i))(x)
        layers.append(x)
    return layers


def decode(res_layers: List[Layer], num_layers: Optional[int] = 15, num_filters: Optional[int] = 64):
    res_layers.reverse()
    x = res_layers[0]
    for i in range(num_layers):
        x = Conv2DTranspose(filters=num_filters,
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            activation='relu',
                            kernel_constraint=min_max_norm(min_value=0., max_value=1.),
                            name='decode{0:d}'.format(i))(x)
        if i % 2 != 0:
            x = Add()([x, res_layers[i]])
    x = Conv2DTranspose(filters=3,
                        kernel_size=3,
                        strides=2,
                        padding='same',
                        activation='sigmoid',
                        name='output')(x)
    x = Add()([x, res_layers[-1]])
    return x


class REDNet30(ConvNet):
    def __init__(self, input_shape: Tuple[int, int, int]):
        super().__init__()
        # ENCODER
        visible = Input(input_shape)
        encode_layers = encode(visible)
        # DECODER
        output = decode(encode_layers)

        self.model = Model(inputs=visible, outputs=output)
