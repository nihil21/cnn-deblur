from models.conv_net import ConvNet
from models.wgan import WGAN, create_patchgan_critic
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Layer, Conv2D, Conv2DTranspose, Add, ELU, ReLU, Lambda,
                                     BatchNormalization, Activation, Reshape)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import logcosh
from typing import Tuple, List, Optional
from utils.custom_metrics import ssim, psnr


def encode(in_layer: Layer,
           num_layers: Optional[int] = 15,
           filters: Optional[int] = 64,
           kernel_size: Optional[int] = 3,
           strides: Optional[int] = 1,
           padding: Optional[str] = 'same',
           use_elu: Optional[bool] = True,
           bn_before_act: Optional[bool] = False) -> List[Layer]:
    layers = []
    x = in_layer
    for i in range(num_layers):
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding=padding,
                   name='encode_conv{:d}'.format(i))(x)
        if bn_before_act:
            x = BatchNormalization(name='encode_bn{:d}'.format(i))(x)
            if use_elu:
                x = ELU(name='encode_act{:d}'.format(i))(x)
            else:
                x = ReLU(name='encode_act{:d}'.format(i))(x)
        else:
            if use_elu:
                x = ELU(name='encode_act{:d}'.format(i))(x)
            else:
                x = ReLU(name='encode_act{:d}'.format(i))(x)
            x = BatchNormalization(name='encode_bn{:d}'.format(i))(x)
        layers.append(x)
    return layers


def decode(res_layers: List[Layer],
           num_layers: Optional[int] = 15,
           filters: Optional[int] = 64,
           kernel_size: Optional[int] = 3,
           strides: Optional[int] = 1,
           padding: Optional[str] = 'same',
           use_elu: Optional[bool] = True,
           bn_before_act: Optional[bool] = False) -> List[Layer]:
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
        if bn_before_act:
            x = BatchNormalization(name='decode_bn{:d}'.format(i))(x)
            if use_elu:
                x = ELU(name='decode_act{:d}'.format(i))(x)
            else:
                x = ReLU(name='decode_act{:d}'.format(i))(x)
        else:
            if use_elu:
                x = ELU(name='decode_act{:d}'.format(i))(x)
            else:
                x = ReLU(name='decode_act{:d}'.format(i))(x)
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


class REDNetV2(Model):
    def __init__(self, input_shape: Tuple[int, int, int],
                 num_layers: Optional[int] = 15):
        super(REDNetV2, self).__init__()
        in_layer = Input(input_shape)

        # Encoder for single channel
        def single_channel_enc(name: str, in_layer_s: Layer):
            x = Conv2D(64,
                       kernel_size=7,
                       strides=2,
                       padding='same',
                       name=f'{name}_enc_conv0')(in_layer_s)
            x = BatchNormalization(name=f'{name}_enc_bn0')(x)
            x = ELU(name=f'{name}_enc_act0')(x)
            layers = [x]
            for i in range(1, num_layers):
                x = Conv2D(64,
                           kernel_size=3,
                           padding='same',
                           name=f'{name}_enc_conv{i}')(x)
                x = BatchNormalization(name=f'{name}_enc_bn{i}')(x)
                x = ELU(name=f'{name}_enc_act{i}')(x)
                layers.append(x)
            return layers

        # Decoder for single channel
        def single_channel_dec(name: str, layers: List[Layer]):
            layers.reverse()
            x = layers[0]
            for i in range(num_layers - 1):
                x = Conv2DTranspose(64,
                                    kernel_size=3,
                                    padding='same',
                                    name=f'{name}_dec_conv{i}')(x)
                x = BatchNormalization(name=f'{name}_dec_bn{i}')(x)
                x = ELU(name=f'{name}_dec_act{i}')(x)
                if i % 2 != 0:
                    x = Add(name=f'{name}_skip_{i-1}')([x, layers[i]])
            x = Conv2DTranspose(1,
                                kernel_size=3,
                                strides=2,
                                padding='same',
                                name=f'{name}_dec_conv16')(x)
            x = BatchNormalization(name=f'{name}_dec_bn16')(x)
            x = ELU(name=f'{name}_dec_act16')(x)

            return x

        # Encoder of red channel
        in_layer_red = Reshape((input_shape[0], input_shape[1], 1))(Lambda(lambda x: x[:, :, :, 0])(in_layer))
        layers_enc_red = single_channel_enc('red', in_layer_red)
        out_layer_red = single_channel_dec('red', layers_enc_red)
        # Encoder of green channel
        in_layer_green = Reshape((input_shape[0], input_shape[1], 1))(Lambda(lambda x: x[:, :, :, 1])(in_layer))
        layers_enc_green = single_channel_enc('green', in_layer_green)
        out_layer_green = single_channel_dec('green', layers_enc_green)
        # Encoder of blue channel
        in_layer_blue = Reshape((input_shape[0], input_shape[1], 1))(Lambda(lambda x: x[:, :, :, 2])(in_layer))
        layers_enc_blue = single_channel_enc('blue', in_layer_blue)
        out_layer_blue = single_channel_dec('blue', layers_enc_blue)

        # Backpropagation is applied to every channel
        self.model = Model(inputs=in_layer,
                           outputs=[out_layer_red, out_layer_green, out_layer_blue])

    def compile(self,
                optimizer='rmsprop',
                loss=None,
                metrics=None,
                loss_weights=None,
                sample_weight_mode=None,
                weighted_metrics=None,
                **kwargs):
        super(REDNetV2, self).compile()

    @tf.function
    def train_step(self, train_batch):
        (blurred_batch, sharp_batch) = train_batch
        sharp_channels = tf.split(sharp_batch, num_or_size_splits=3, axis=3)
        with tf.GradientTape() as tape:
            restored = self.model(blurred_batch, training=True)
            loss_red = self.loss(sharp_channels[0], restored[0])
            loss_green = self.loss(sharp_channels[1], restored[1])
            loss_blue = self.loss(sharp_channels[2], restored[2])
            loss_value = tf.reduce_mean(loss_red, loss_blue, loss_green)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        merged = tf.concat(restored, axis=3)
        ssim_metric = ssim(sharp_batch, merged)
        psnr_metric = psnr(sharp_batch, merged)
        return {'loss': loss_value,
                'ssim': ssim_metric,
                'psnr': psnr_metric}


class REDNet30WGAN(WGAN):
    def __init__(self, input_shape: Tuple[int, int, int]):
        # ENCODER
        visible = Input(input_shape)
        encode_layers = encode(visible, use_elu=False)
        # DECODER
        decode_layers = decode(encode_layers, use_elu=False)
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
        def generator_loss(sharp_batch: tf.Tensor,
                           predicted_batch: tf.Tensor,
                           fake_logits: tf.Tensor):
            adv_loss = -tf.reduce_mean(fake_logits)
            content_loss = tf.reduce_mean(logcosh(sharp_batch, predicted_batch))
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
