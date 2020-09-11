from models.conv_net import ConvNet
from models.wgan import WGAN, create_patchgan_critic
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Conv2D, Conv2DTranspose, Add, ELU, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import logcosh
from typing import Tuple, List, Optional


def encode(in_layer: Layer,
           num_layers: Optional[int] = 15,
           filters: Optional[int] = 64,
           kernel_size: Optional[int] = 3,
           strides: Optional[int] = 1,
           padding: Optional[str] = 'same') -> List[Layer]:
    layers = []
    x = in_layer
    for i in range(num_layers):
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding=padding,
                   name='encode_conv{:d}'.format(i))(x)
        x = ELU(name='encode_act{:d}'.format(i))(x)
        x = BatchNormalization(name='encode_bn{:d}'.format(i))(x)
        layers.append(x)
    return layers


def decode(res_layers: List[Layer],
           num_layers: Optional[int] = 15,
           filters: Optional[int] = 64,
           kernel_size: Optional[int] = 3,
           strides: Optional[int] = 1,
           padding: Optional[str] = 'same') -> List[Layer]:
    layers = []
    res_layers.reverse()
    x = res_layers[0]
    for i in range(num_layers):
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding=padding,
                   name='decode_conv{:d}'.format(i))(x)
        if i % 2 != 0:
            x = Add(name='decode_skip{:d}'.format(i))([x, res_layers[i]])
        x = ELU(name='decode_act{:d}'.format(i))(x)
        x = BatchNormalization(name='decode_bn{:d}'.format(i))(x)
        layers.append(x)

    return layers


class REDNet10(ConvNet):
    def __init__(self, input_shape: Tuple[int, int, int]):
        super(REDNet10, self).__init__()
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
        super(REDNet20, self).__init__()
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
        super(REDNet30, self).__init__()
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


class REDNet30WGAN(WGAN):
    def __init__(self, input_shape: Tuple[int, int, int]):
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
        output = Activation('tanh')(output)

        # Create generator and critic models
        generator = Model(inputs=visible, outputs=output)
        critic = create_patchgan_critic(input_shape)

        # Set critic_updates, i.e. the times the critic gets trained w.r.t. one training step of the generator
        self.critic_updates = 5
        # Set weight of gradient penalty
        self.gp_weight = 10.0

        # Define and set loss functions
        def generator_loss(sharp_pyramid: tf.Tensor,
                           predicted_pyramid: tf.Tensor,
                           fake_logits: tf.Tensor):
            adv_loss = -tf.reduce_mean(fake_logits)
            content_loss = logcosh(sharp_pyramid, predicted_pyramid)
            return content_loss + 1e-4 * adv_loss

        def critic_loss(real_logits: tf.Tensor,
                        fake_logits: tf.Tensor):
            return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

        # Set optimizers as Adam with given learning_rate
        g_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.9)
        c_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.9)

        # Call base-class init method
        super(REDNet30WGAN, self).__init__(generator,
                                           critic,
                                           generator_loss,
                                           critic_loss,
                                           g_optimizer,
                                           c_optimizer)
