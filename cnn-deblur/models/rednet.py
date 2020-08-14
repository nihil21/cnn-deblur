from models.conv_net import ConvNet
from tensorflow.keras.layers import Input, Layer, Conv2D, Conv2DTranspose, Add, ELU, BatchNormalization, concatenate
from tensorflow.keras.models import Model
from typing import Tuple, List, Optional


def encode(in_layer: Layer,
           num_layers: Optional[int] = 15,
           filters: Optional[int] = 64,
           kernel_size: Optional[int] = 3,
           strides: Optional[int] = 1,
           padding: Optional[str] = 'same',
           scale_id: Optional[str] = '') -> List[Layer]:
    layers = []
    x = in_layer
    for i in range(num_layers):
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding=padding,
                   name='encode_conv{:d}{:s}'.format(i, scale_id))(x)
        x = ELU(name='encode_act{:d}{:s}'.format(i, scale_id))(x)
        x = BatchNormalization(name='encode_bn{:d}{:s}'.format(i, scale_id))(x)
        layers.append(x)
    return layers


def decode(res_layers: List[Layer],
           num_layers: Optional[int] = 15,
           filters: Optional[int] = 64,
           kernel_size: Optional[int] = 3,
           strides: Optional[int] = 1,
           padding: Optional[str] = 'same',
           scale_id: Optional[str] = '') -> List[Layer]:
    layers = []
    res_layers.reverse()
    x = res_layers[0]
    for i in range(num_layers):
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding=padding,
                   name='decode_conv{:d}{:s}'.format(i, scale_id))(x)
        if i % 2 != 0:
            x = Add(name='decode_skip{:d}{:s}'.format(i, scale_id))([x, res_layers[i]])
        x = ELU(name='decode_act{:d}{:s}'.format(i, scale_id))(x)
        x = BatchNormalization(name='decode_bn{:d}{:s}'.format(i, scale_id))(x)
        layers.append(x)

    return layers


"""class ConvBlock(Layer):
    def __init__(self,
                 name: str,
                 filters: int,
                 kernel_size: int = 3,
                 strides: int = 1,
                 padding: str = 'same',
                 activation: str = 'elu',
                 use_transpose: bool = False,
                 res_layer: Layer = None,
                 **kwargs):
        super().__init__(name, **kwargs)
        self.__conv_layer = Conv2D(filters=filters,
                                   kernel_size=kernel_size,
                                   strides=strides,
                                   padding=padding) if not use_transpose else Conv2DTranspose(filters=filters,
                                                                                              kernel_size=kernel_size,
                                                                                              strides=strides,
                                                                                              padding=padding)
        self.__res_layer = res_layer
        self.__activation = ELU() if activation == 'elu' else ReLU()
        self.__bn = BatchNormalization()

    def call(self, inputs, **kwargs):
        x = self.__conv_layer(inputs)
        if self.__res_layer is not None:
            x = Add()([x, self.__res_layer])
        x = self.__activation(x)
        return self.__bn(x)"""


class REDNet10(ConvNet):
    def __init__(self, input_shape: Tuple[int, int, int]):
        super().__init__()
        # ENCODER
        visible = Input(input_shape)
        encode_layers = encode(visible, num_layers=5)
        # DECODER
        decode_layers = decode(encode_layers, num_layers=5)
        output = Conv2DTranspose(filters=3,
                                 kernel_size=1,
                                 strides=1,
                                 padding='same',
                                 name='output_conv')(decode_layers[-1])
        output = Add(name='output_skip')([output, visible])
        output = ELU(name='output_elu')(output)

        self.model = Model(inputs=visible, outputs=output)


class REDNet20(ConvNet):
    def __init__(self, input_shape: Tuple[int, int, int]):
        super().__init__()
        # ENCODER
        visible = Input(input_shape)
        encode_layers = encode(visible, num_layers=10)
        # DECODER
        decode_layers = decode(encode_layers, num_layers=10)
        output = Conv2DTranspose(filters=3,
                                 kernel_size=1,
                                 strides=1,
                                 padding='same',
                                 name='output_conv')(decode_layers[-1])
        output = Add(name='output_skip')([output, visible])
        output = ELU(name='output_elu')(output)

        self.model = Model(inputs=visible, outputs=output)


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
                                 name='output_conv')(decode_layers[-1])
        output = Add(name='output_skip')([output, visible])
        output = ELU(name='output_elu')(output)

        self.model = Model(inputs=visible, outputs=output)


class MSREDNet30(ConvNet):
    def __init__(self, input_shape: Tuple[int, int, int]):
        super().__init__()
        # --- Coarsest branch ---
        # ENCODER
        in_layer3 = Input(shape=(input_shape[0] // 4, input_shape[1] // 4, input_shape[2]),
                          name='in_layer3')
        encode_layers3 = encode(in_layer3, scale_id='3')
        # DECODER
        decode_layers3 = decode(encode_layers3, scale_id='3')
        out_layer3 = Conv2DTranspose(filters=3,
                                     kernel_size=1,
                                     strides=1,
                                     padding='same',
                                     name='output_conv3')(decode_layers3[-1])
        out_layer3 = Add(name='output_skip3')([out_layer3, in_layer3])
        out_layer3 = ELU(name='output_elu3')(out_layer3)
        # --- Middle branch ---
        # ENCODER
        in_layer2 = Input(shape=(input_shape[0] // 2, input_shape[1] // 2, input_shape[2]),
                          name='in_layer2')
        up_conv2 = Conv2DTranspose(filters=64,
                                   kernel_size=5,
                                   strides=2,
                                   padding='same')(out_layer3)
        concat2 = concatenate([in_layer2, up_conv2])
        encode_layers2 = encode(concat2, scale_id='2')
        # DECODER
        decode_layers2 = decode(encode_layers2, scale_id='2')
        out_layer2 = Conv2DTranspose(filters=3,
                                     kernel_size=1,
                                     strides=1,
                                     padding='same',
                                     name='output_conv2')(decode_layers2[-1])
        out_layer2 = Add(name='output_skip2')([out_layer2, in_layer2])
        out_layer2 = ELU(name='output_elu2')(out_layer2)
        # --- Finest branch ---
        # ENCODER
        in_layer1 = Input(shape=input_shape,
                          name='in_layer1')
        up_conv1 = Conv2DTranspose(filters=64,
                                   kernel_size=5,
                                   strides=2,
                                   padding='same')(out_layer2)
        concat1 = concatenate([in_layer1, up_conv1])
        encode_layers1 = encode(concat1, scale_id='1')
        # DECODER
        decode_layers1 = decode(encode_layers1, scale_id='1')
        out_layer1 = Conv2DTranspose(filters=3,
                                     kernel_size=1,
                                     strides=1,
                                     padding='same',
                                     name='output_conv1')(decode_layers1[-1])
        out_layer1 = Add(name='output_skip1')([out_layer1, in_layer1])
        out_layer1 = ELU(name='output_elu1')(out_layer1)

        self.model = Model(inputs=[in_layer1, in_layer2, in_layer3],
                           outputs=[out_layer1, out_layer2, out_layer3])
