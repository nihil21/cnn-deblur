import typing

import tensorflow as tf
from tensorflow import keras

from .conv_net import ConvNet
from .wgan import WGAN, create_patchgan_critic


def encode(
        in_layer: keras.layers.Layer,
        num_layers: int = 15,
        filters: int = 64,
        kernel_size: int = 3,
        strides: int = 1,
        padding: str = 'same',
        use_elu: bool = True,
        bn_before_act: bool = False
) -> typing.List[keras.layers.Layer]:
    layers = []
    x = in_layer
    for i in range(num_layers):
        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            name='encode_conv{:d}'.format(i)
        )(x)
        if bn_before_act:
            x = keras.layers.BatchNormalization(name='encode_bn{:d}'.format(i))(x)
            if use_elu:
                x = keras.layers.ELU(name='encode_act{:d}'.format(i))(x)
            else:
                x = keras.layers.ReLU(name='encode_act{:d}'.format(i))(x)
        else:
            if use_elu:
                x = keras.layers.ELU(name='encode_act{:d}'.format(i))(x)
            else:
                x = keras.layers.ReLU(name='encode_act{:d}'.format(i))(x)
            x = keras.layers.BatchNormalization(name='encode_bn{:d}'.format(i))(x)
        layers.append(x)
    return layers


def decode(
        res_layers: typing.List[keras.layers.Layer],
        num_layers: int = 15,
        filters: int = 64,
        kernel_size: int = 3,
        strides: int = 1,
        padding: str = 'same',
        use_elu: bool = True,
        bn_before_act: bool = False
) -> typing.List[keras.layers.Layer]:
    layers = []
    res_layers.reverse()
    x = res_layers[0]
    for i in range(num_layers):
        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            name='decode_conv{:d}'.format(i)
        )(x)
        if i % 2 != 0:
            x = keras.layers.Add(name='decode_skip{:d}'.format(i))([x, res_layers[i]])
        if bn_before_act:
            x = keras.layers.BatchNormalization(name='decode_bn{:d}'.format(i))(x)
            if use_elu:
                x = keras.layers.ELU(name='decode_act{:d}'.format(i))(x)
            else:
                x = keras.layers.ReLU(name='decode_act{:d}'.format(i))(x)
        else:
            if use_elu:
                x = keras.layers.ELU(name='decode_act{:d}'.format(i))(x)
            else:
                x = keras.layers.ReLU(name='decode_act{:d}'.format(i))(x)
            x = keras.layers.BatchNormalization(name='decode_bn{:d}'.format(i))(x)
        layers.append(x)

    return layers


class REDNet10(ConvNet):
    def __init__(self, input_shape: typing.Tuple[int, int, int]):
        super(REDNet10, self).__init__()
        # ENCODER
        visible = keras.layers.Input(input_shape)
        encode_layers = encode(visible, num_layers=5)
        # DECODER
        decode_layers = decode(encode_layers, num_layers=5)
        output = keras.layers.Conv2DTranspose(
            filters=3,
            kernel_size=1,
            strides=1,
            padding='same',
            name='output_conv'
        )(decode_layers[-1])
        output = keras.layers.Add(name='output_skip')([output, visible])
        output = keras.layers.ELU(name='output_elu')(output)

        self.model = keras.models.Model(inputs=visible, outputs=output)


class REDNet20(ConvNet):
    def __init__(self, input_shape: typing.Tuple[int, int, int]):
        super(REDNet20, self).__init__()
        # ENCODER
        visible = keras.layers.Input(input_shape)
        encode_layers = encode(visible, num_layers=10)
        # DECODER
        decode_layers = decode(encode_layers, num_layers=10)
        output = keras.layers.Conv2DTranspose(
            filters=3,
            kernel_size=1,
            strides=1,
            padding='same',
            name='output_conv'
        )(decode_layers[-1])
        output = keras.layers.Add(name='output_skip')([output, visible])
        output = keras.layers.ELU(name='output_elu')(output)

        self.model = keras.models.Model(inputs=visible, outputs=output)


class REDNet30(ConvNet):
    def __init__(
            self,
            input_shape: typing.Tuple[int, int, int],
            bn_before_act: bool = False
    ):
        super(REDNet30, self).__init__()
        # ENCODER
        visible = keras.layers.Input(input_shape)
        encode_layers = encode(visible,
                               bn_before_act=bn_before_act)
        # DECODER
        decode_layers = decode(encode_layers,
                               bn_before_act=bn_before_act)
        output = keras.layers.Conv2DTranspose(
            filters=3,
            kernel_size=1,
            strides=1,
            padding='same',
            name='output_conv'
        )(decode_layers[-1])
        output = keras.layers.Add(name='output_skip')([output, visible])
        output = keras.layers.ELU(name='output_elu')(output)

        self.model = keras.models.Model(inputs=visible, outputs=output)


class REDNet30WGAN(WGAN):
    def __init__(
            self,
            input_shape: typing.Tuple[int, int, int],
            use_elu: bool = False
    ):
        # ENCODER
        visible = keras.layers.Input(input_shape)
        encode_layers = encode(visible, use_elu=use_elu)
        # DECODER
        decode_layers = decode(encode_layers, use_elu=use_elu)
        output = keras.layers.Conv2DTranspose(
            filters=3,
            kernel_size=1,
            strides=1,
            padding='same',
            name='output_conv'
        )(decode_layers[-1])
        output = keras.layers.Add(name='output_skip')([output, visible])
        output = keras.layers.Activation('tanh')(output)

        # Create generator and critic models
        generator = keras.models.Model(inputs=visible, outputs=output)
        critic = create_patchgan_critic(input_shape)

        # Set critic_updates, i.e. the times the critic gets trained w.r.t. one training step of the generator
        self.critic_updates = 5
        # Set weight of gradient penalty
        self.gp_weight = 10.0

        # Define and set loss functions
        def generator_loss(
                sharp_batch: tf.Tensor,
                predicted_batch: tf.Tensor,
                fake_logits: tf.Tensor
        ):
            adv_loss = -tf.reduce_mean(fake_logits)
            content_loss = tf.reduce_mean(keras.losses.logcosh(sharp_batch, predicted_batch))
            return content_loss + 1e-4 * adv_loss

        def critic_loss(real_logits: tf.Tensor,
                        fake_logits: tf.Tensor):
            return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

        # Set optimizers as Adam with given learning_rate
        g_optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.9)
        c_optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.9)

        # Call base-class init method
        super(REDNet30WGAN, self).__init__(
            generator,
            critic,
            generator_loss,
            critic_loss,
            g_optimizer,
            c_optimizer
        )
