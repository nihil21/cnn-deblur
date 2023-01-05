import typing

import tensorflow as tf
from tensorflow import keras

from ..utils.custom_losses import ms_logcosh
from .wgan import WGAN, create_patchgan_critic


def res_block(in_layer: keras.layers.Layer,
              layer_id: str,
              filters: int = 64,
              kernels: int = 5,
              use_batchnorm: bool = True,
              use_elu: bool = False,
              last_act: bool = False):
    # Block 1
    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernels,
        padding='same',
        name='res_conv{:s}_1'.format(layer_id)
    )(in_layer)
    if use_batchnorm:
        x = keras.layers.BatchNormalization(name='res_bn{:s}_1'.format(layer_id))(x)
    if use_elu:
        x = keras.layers.ELU(name='res_elu{:s}_1'.format(layer_id))(x)
    else:
        x = keras.layers.ReLU(name='res_relu{:s}_1'.format(layer_id))(x)
    # Block 2
    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernels,
        padding='same',
        name='res_conv{:s}_2'.format(layer_id)
    )(x)
    if use_batchnorm:
        x = keras.layers.BatchNormalization(name='res_bn{:s}_2'.format(layer_id))(x)
    # Skip connection
    x = keras.layers.Add(name='res_add{:s}'.format(layer_id))([x, in_layer])
    if last_act:
        if use_elu:
            x = keras.layers.ELU(name='res_elu{:s}_2'.format(layer_id))(x)
        else:
            x = keras.layers.ReLU(name='res_relu{:s}_2'.format(layer_id))(x)
    return x


def create_generator(
        input_shape,
        use_elu: bool = False,
        last_act: bool = False,
        num_res_blocks: int = 19
):
    height = input_shape[0]
    width = input_shape[1]
    in_layer1 = keras.layers.Input(shape=input_shape)
    # Coarsest branch
    in_layer3 = keras.layers.Lambda(lambda t: tf.image.resize(t, size=(height // 4, width // 4)))(in_layer1)
    # in_layer3 = Input(shape=(input_shape[0] // 4, input_shape[1] // 4, input_shape[2]),
    #                   name='in_layer3')
    conv3 = keras.layers.Conv2D(
        filters=64,
        kernel_size=5,
        padding='same',
        name='conv3'
    )(in_layer3)
    x = conv3
    for i in range(num_res_blocks):
        x = res_block(
            in_layer=x,
            layer_id='3_{:d}'.format(i),
            use_elu=use_elu,
            last_act=last_act
        )
    out_layer3 = keras.layers.Conv2D(
        filters=3,
        kernel_size=5,
        padding='same',
        activation='tanh',
        name='out_layer_3'
    )(x)

    # Middle branch
    in_layer2 = keras.layers.Lambda(lambda t: tf.image.resize(t, size=(height // 2, width // 2)))(in_layer1)
    # in_layer2 = Input(shape=(input_shape[0] // 2, input_shape[1] // 2, input_shape[2]),
    #                   name='in_layer2')
    up_conv2 = keras.layers.Conv2DTranspose(
        filters=61,
        kernel_size=5,
        strides=2,
        padding='same'
    )(out_layer3)
    concat2 = keras.layers.Concatenate(axis=3)([in_layer2, up_conv2])
    conv2 = keras.layers.Conv2D(
        filters=64,
        kernel_size=5,
        padding='same',
        name='conv2'
    )(concat2)
    x = conv2
    for i in range(num_res_blocks):
        x = res_block(
            in_layer=x,
            layer_id='2_{:d}'.format(i),
            last_act=last_act,
            use_elu=use_elu
        )
    out_layer2 = keras.layers.Conv2D(
        filters=3,
        kernel_size=5,
        padding='same',
        activation='tanh',
        name='out_layer2'
    )(x)

    # Finest branch
    up_conv1 = keras.layers.Conv2DTranspose(
        filters=61,
        kernel_size=5,
        strides=2,
        padding='same'
    )(out_layer2)
    concat1 = keras.layers.Concatenate(axis=3)([in_layer1, up_conv1])
    conv1 = keras.layers.Conv2D(
        filters=64,
        kernel_size=5,
        padding='same',
        name='conv1'
    )(concat1)
    x = conv1
    for i in range(num_res_blocks):
        x = res_block(
            in_layer=x,
            layer_id='1_{:d}'.format(i),
            last_act=last_act,
            use_elu=use_elu
        )
    out_layer1 = keras.layers.Conv2D(
        filters=3,
        kernel_size=5,
        padding='same',
        activation='tanh',
        name='out_layer1'
    )(x)

    # Final model
    generator = keras.models.Model(
        inputs=in_layer1,
        outputs=[out_layer1, out_layer2, out_layer3],
        name='Generator'
    )
    return generator


class MSDeblurWGAN(WGAN):
    def __init__(
            self,
            input_shape: typing.Tuple[int, int, int],
            use_elu: bool = False,
            use_sigmoid: bool = False,
            use_bn: bool = False,
            last_act: bool = False,
            num_res_blocks: int = 19
    ):
        # Build generator and critic
        generator = create_generator(
            input_shape,
            use_elu,
            last_act,
            num_res_blocks
        )
        critic = create_patchgan_critic(
            input_shape,
            use_elu,
            use_sigmoid,
            use_bn
        )

        # Define and set loss functions
        def generator_loss(
                sharp_pyramid: typing.List[tf.Tensor],
                predicted_pyramid: typing.List[tf.Tensor],
                fake_logits: tf.Tensor
        ):
            adv_loss = -tf.reduce_mean(fake_logits)
            content_loss = ms_logcosh(sharp_pyramid, predicted_pyramid)
            return content_loss + 1e-4 * adv_loss

        def critic_loss(
                real_logits: tf.Tensor,
                fake_logits: tf.Tensor
        ):
            return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

        # Set optimizers as Adam with given learning_rate
        g_optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.9)
        c_optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.9)

        # Call base-class init method
        super(MSDeblurWGAN, self).__init__(
            generator,
            critic,
            generator_loss,
            critic_loss,
            g_optimizer,
            c_optimizer
        )

    # Override train_step and test_step in order to account for pyramids instead of single-scale images
    @tf.function
    def train_step(
            self,
            train_batch: typing.Tuple[tf.Tensor, tf.Tensor]
    ):
        # Determine batch size, height and width
        batch_size = tf.shape(train_batch[0])[0]
        height = tf.shape(train_batch[0])[1]
        width = tf.shape(train_batch[0])[2]
        # Prepare Gaussian pyramid
        blurred_batch = train_batch[0]
        sharp_batch1 = train_batch[1]
        # blurred_batch2 = tf.image.resize(train_batch[0], size=(height // 2, width // 2))
        sharp_batch2 = tf.image.resize(train_batch[1], size=(height // 2, width // 2))
        # blurred_batch3 = tf.image.resize(train_batch[0], size=(height // 4, width // 4))
        sharp_batch3 = tf.image.resize(train_batch[1], size=(height // 4, width // 4))
        # blurred_pyramid = [blurred_batch1, blurred_batch2, blurred_batch3]
        sharp_pyramid = [sharp_batch1, sharp_batch2, sharp_batch3]

        c_losses = []
        # Train the critic multiple times according to critic_updates (by default, 5)
        for _ in range(self.critic_updates):
            with tf.GradientTape() as c_tape:
                # Make predictions
                predicted_pyramid = self.generator(blurred_batch, training=True)
                # Get logits for both fake and real images (only original scale)
                fake_logits = self.critic(predicted_pyramid[0], training=True)
                real_logits = self.critic(sharp_pyramid[0], training=True)
                # Calculate critic's loss
                c_loss = self.c_loss(real_logits, fake_logits)
                # Calculate gradient penalty
                gp = self.gradient_penalty(
                    batch_size,
                    real_imgs=tf.cast(sharp_pyramid[0], dtype='float32'),
                    fake_imgs=predicted_pyramid[0]
                )
                # Add gradient penalty to the loss
                c_loss += gp * self.gp_weight
            # Get gradient w.r.t. critic's loss and update weights
            c_grad = c_tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(c_grad, self.critic.trainable_variables))

            c_losses.append(c_loss)

        # Train the generator
        with tf.GradientTape() as g_tape:
            # Make predictions
            predicted_pyramid = self.generator(blurred_batch, training=True)
            # Get logits for fake images (only original scale)
            fake_logits = self.critic(predicted_pyramid[0], training=True)
            # Calculate generator's loss
            g_loss = self.g_loss(sharp_pyramid, predicted_pyramid, fake_logits)
        # Get gradient w.r.t. generator's loss and update weights
        g_grad = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grad, self.generator.trainable_variables))

        # Compute metrics
        ssim_metric = tf.image.ssim(
            sharp_pyramid[0],
            tf.cast(predicted_pyramid[0], dtype='bfloat16'),
            max_val=2.
        )
        psnr_metric = tf.image.psnr(
            sharp_pyramid[0],
            tf.cast(predicted_pyramid[0], dtype='bfloat16'),
            max_val=2.
        )

        return {
            'g_loss': g_loss,
            'ssim': tf.reduce_mean(ssim_metric),
            'psnr': tf.reduce_mean(psnr_metric),
            'c_loss': tf.reduce_mean(c_losses)
        }

    @tf.function
    def test_step(
            self,
            val_batch: typing.Tuple[tf.Tensor, tf.Tensor]
    ):
        # Determine height and width
        height = tf.shape(val_batch[0])[1]
        width = tf.shape(val_batch[0])[2]
        # Prepare Gaussian pyramid
        blurred_batch = val_batch[0]
        sharp_batch1 = val_batch[1]
        # blurred_batch2 = tf.image.resize(val_batch[0], size=(height // 2, width // 2))
        sharp_batch2 = tf.image.resize(val_batch[1], size=(height // 2, width // 2))
        # blurred_batch3 = tf.image.resize(val_batch[0], size=(height // 4, width // 4))
        sharp_batch3 = tf.image.resize(val_batch[1], size=(height // 4, width // 4))
        # blurred_pyramid = [blurred_batch1, blurred_batch2, blurred_batch3]
        sharp_pyramid = [sharp_batch1, sharp_batch2, sharp_batch3]

        # Generate fake inputs
        predicted_pyramid = self.generator(blurred_batch, training=False)
        # Get logits for both fake and real images
        fake_logits = self.critic(sharp_pyramid, training=False)
        real_logits = self.critic(predicted_pyramid, training=False)
        # Calculate critic's loss
        c_loss = self.c_loss(real_logits, fake_logits)
        # Calculate generator's loss
        g_loss = self.g_loss(sharp_pyramid, predicted_pyramid, fake_logits)

        # Compute metrics
        ssim_metric = tf.image.ssim(
            sharp_pyramid[0],
            tf.cast(predicted_pyramid[0], dtype='bfloat16'),
            max_val=2.
        )
        psnr_metric = tf.image.psnr(
            sharp_pyramid[0],
            tf.cast(predicted_pyramid[0], dtype='bfloat16'),
            max_val=2.
        )

        return {
            'g_loss': g_loss,
            'ssim': tf.reduce_mean(ssim_metric),
            'psnr': tf.reduce_mean(psnr_metric),
            'c_loss': c_loss
        }
