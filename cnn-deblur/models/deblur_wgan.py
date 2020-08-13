from models.wgan import WGAN, create_patchgan_critic
from utils.custom_losses import perceptual_loss
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Conv2D, Conv2DTranspose, Dropout, Add, ELU, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from typing import Tuple, Optional


# Function to build generator and critic
def res_block(in_layer: Layer,
              layer_id: int,
              filters: Optional[int] = 256,
              kernel_size: Optional[Tuple[int]] = 3,
              use_elu: Optional[bool] = False,
              use_dropout: Optional[bool] = False):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',
               name='res_block_conv{:d}_1'.format(layer_id))(in_layer)
    x = tfa.layers.InstanceNormalization(name='res_block_in{:d}_1'.format(layer_id))(x)
    if use_elu:
        x = ELU(name='res_block_elu{:d}'.format(layer_id))(x)
    else:
        x = ReLU(name='res_block_relu{:d}'.format(layer_id))(x)
    if use_dropout:
        x = Dropout(rate=0.5,
                    name='res_block_drop{:d}'.format(layer_id))(x)
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',
               name='res_block_conv{:d}_2'.format(layer_id))(x)
    x = tfa.layers.InstanceNormalization(name='res_block_in{:d}_2'.format(layer_id))(x)
    x = Add(name='res_block_add{:d}'.format(layer_id))([x, in_layer])
    return x


def create_generator(input_shape,
                     use_elu: Optional[bool] = False,
                     use_dropout: Optional[bool] = False,
                     num_res_blocks: Optional[int] = 9):
    in_layer = Input(input_shape)
    # Block 1
    x = Conv2D(filters=64,
               kernel_size=7,
               padding='same',
               name='conv1')(in_layer)
    x = tfa.layers.InstanceNormalization(name='in1')(x)
    if use_elu:
        x = ELU(name='elu1')(x)
    else:
        x = ReLU(name='relu1')(x)
    # Block 2
    x = Conv2D(filters=128,
               kernel_size=3,
               strides=2,
               padding='same',
               name='conv2')(x)
    x = tfa.layers.InstanceNormalization(name='in2')(x)
    if use_elu:
        x = ELU(name='elu2')(x)
    else:
        x = ReLU(name='relu2')(x)
    # Block 3
    x = Conv2D(filters=256,
               kernel_size=3,
               strides=2,
               padding='same',
               name='conv3')(x)
    x = tfa.layers.InstanceNormalization(name='in3')(x)
    if use_elu:
        x = ELU(name='elu3')(x)
    else:
        x = ReLU(name='relu3')(x)
    # ResBlocks
    for i in range(num_res_blocks):
        x = res_block(in_layer=x,
                      layer_id=i,
                      use_elu=use_elu,
                      use_dropout=use_dropout)
    # Block 4
    x = Conv2DTranspose(filters=128,
                        kernel_size=3,
                        strides=2,
                        padding='same',
                        name='conv4')(x)
    x = tfa.layers.InstanceNormalization(name='in4')(x)
    if use_elu:
        x = ELU(name='elu4')(x)
    else:
        x = ReLU(name='relu4')(x)
    # Block 5
    x = Conv2DTranspose(filters=64,
                        kernel_size=3,
                        strides=2,
                        padding='same',
                        name='conv5')(x)
    x = tfa.layers.InstanceNormalization(name='in5')(x)
    if use_elu:
        x = ELU(name='elu5')(x)
    else:
        x = ReLU(name='relu5')(x)
    # Block 6
    x = Conv2D(filters=3,
               kernel_size=7,
               padding='same',
               activation='tanh',
               name='conv6')(x)
    out_layer = Add(name='add')([x, in_layer])

    generator = Model(inputs=in_layer, outputs=out_layer, name='Generator')
    return generator


class DeblurWGan(WGAN):
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 use_elu: Optional[bool] = False,
                 use_dropout: Optional[bool] = False,
                 num_res_blocks: Optional[int] = 9):
        # Build generator and critic
        generator = create_generator(input_shape,
                                     use_elu,
                                     use_dropout,
                                     num_res_blocks)
        critic = create_patchgan_critic(input_shape,
                                        use_elu)

        # Set loss_model, based on VGG16, to compute perceptual loss
        vgg = VGG16(include_top=False, weights='imagenet', input_shape=(None, None, 3))
        loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
        loss_model.trainable = False

        # Define and set loss functions
        def generator_loss(sharp_batch: tf.Tensor,
                           generated_batch: tf.Tensor,
                           fake_logits: tf.Tensor):
            adv_loss = tf.reduce_mean(-fake_logits)
            content_loss = perceptual_loss(sharp_batch, generated_batch, loss_model)
            return adv_loss + 100.0 * content_loss

        def critic_loss(real_logits: tf.Tensor,
                        fake_logits: tf.Tensor):
            return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

        # Set optimizers as Adam with lr=1e-4
        g_optimizer = Adam(lr=1e-4)
        c_optimizer = Adam(lr=1e-4)

        super(DeblurWGan, self).__init__(generator, critic, generator_loss, critic_loss, g_optimizer, c_optimizer)
