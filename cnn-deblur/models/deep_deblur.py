import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Layer, Conv2D, Conv2DTranspose, Add, Dense,
                                     ELU, ReLU, LeakyReLU, BatchNormalization, concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.losses import logcosh
from tensorflow.keras.optimizers import Adam
from utils.custom_metrics import ssim, psnr
import operator
import functools
from tqdm import notebook
from typing import Tuple, List, Optional


def res_block(in_layer: Layer,
              layer_id: str,
              filters: Optional[int] = 64,
              kernels: Optional[int] = 5,
              use_batchnorm: Optional[bool] = True,
              use_elu: Optional[bool] = False):
    # Block 1
    x = Conv2D(filters=filters,
               kernel_size=kernels,
               padding='same',
               name='res_conv{:s}_1'.format(layer_id))(in_layer)
    if use_batchnorm:
        x = BatchNormalization(name='res_bn{:s}_1'.format(layer_id))(x)
    if use_elu:
        x = ELU(name='res_elu{:s}_1'.format(layer_id))(x)
    else:
        x = ReLU(name='res_relu{:s}_1'.format(layer_id))(x)
    # Block 2
    x = Conv2D(filters=filters,
               kernel_size=kernels,
               padding='same',
               name='res_conv{:s}_2'.format(layer_id))(x)
    if use_batchnorm:
        x = BatchNormalization(name='res_bn{:s}_2'.format(layer_id))(x)
    # Skip connection
    x = Add(name='res_add{:s}'.format(layer_id))([x, in_layer])
    if use_elu:
        x = ELU(name='res_elu{:s}_2'.format(layer_id))(x)
    else:
        x = ReLU(name='res_relu{:s}_2'.format(layer_id))(x)
    return x


def create_generator(input_shape,
                     use_elu: Optional[bool] = False,
                     num_res_blocks: Optional[int] = 19):
    # Coarsest branch
    in_layer3 = Input(shape=(input_shape[0] // 4, input_shape[1] // 4, input_shape[2]),
                      name='in_layer3')
    conv3 = Conv2D(filters=64,
                   kernel_size=5,
                   padding='same',
                   name='conv3')(in_layer3)
    x = conv3
    for i in range(num_res_blocks):
        x = res_block(in_layer=x,
                      layer_id='3_{:d}'.format(i),
                      use_elu=use_elu)
    out_layer3 = Conv2D(filters=3,
                        kernel_size=5,
                        padding='same',
                        name='out_layer_3')(x)

    # Middle branch
    in_layer2 = Input(shape=(input_shape[0] // 2, input_shape[1] // 2, input_shape[2]),
                      name='in_layer2')
    up_conv2 = Conv2DTranspose(filters=64,
                               kernel_size=5,
                               strides=2,
                               padding='same')(out_layer3)
    concat2 = concatenate([in_layer2, up_conv2])
    conv2 = Conv2D(filters=64,
                   kernel_size=5,
                   padding='same',
                   name='conv2')(concat2)
    x = conv2
    for i in range(num_res_blocks):
        x = res_block(in_layer=x,
                      layer_id='2_{:d}'.format(i),
                      use_elu=use_elu)
    out_layer2 = Conv2D(filters=3,
                        kernel_size=5,
                        padding='same',
                        name='out_layer2')(x)

    # Finest branch
    in_layer1 = Input(shape=input_shape,
                      name='in_layer1')
    up_conv1 = Conv2DTranspose(filters=64,
                               kernel_size=5,
                               strides=2,
                               padding='same')(out_layer2)
    concat1 = concatenate([in_layer1, up_conv1])
    conv1 = Conv2D(filters=64,
                   kernel_size=5,
                   padding='same',
                   name='conv1')(concat1)
    x = conv1
    for i in range(num_res_blocks):
        x = res_block(in_layer=x,
                      layer_id='1_{:d}'.format(i),
                      use_elu=use_elu)
    out_layer1 = Conv2D(filters=3,
                        kernel_size=5,
                        padding='same',
                        name='out_layer1')(x)

    # Final model
    generator = Model(inputs=[in_layer1, in_layer2, in_layer3],
                      outputs=[out_layer1, out_layer2, out_layer3],
                      name='Generator')
    return generator


def create_discriminator(input_shape,
                         use_elu: Optional[bool] = False):
    in_layer = Input(input_shape)
    # Block 1
    x = Conv2D(filters=32,
               kernel_size=5,
               strides=2,
               padding='same',
               name='conv1')(in_layer)
    if use_elu:
        x = ELU(name='elu1')(x)
    else:
        x = LeakyReLU(name='lrelu1')(x)
    # Block 2
    x = Conv2D(filters=64,
               kernel_size=5,
               strides=1,
               padding='same',
               name='conv2')(x)
    if use_elu:
        x = ELU(name='elu2')(x)
    else:
        x = LeakyReLU(name='lrelu2')(x)
    # Block 3
    x = Conv2D(filters=64,
               kernel_size=5,
               strides=2,
               padding='same',
               name='conv3')(x)
    if use_elu:
        x = ELU(name='elu3')(x)
    else:
        x = LeakyReLU(name='lrelu3')(x)
    # Block 4
    x = Conv2D(filters=128,
               kernel_size=5,
               strides=1,
               padding='same',
               name='conv4')(x)
    if use_elu:
        x = ELU(name='elu4')(x)
    else:
        x = LeakyReLU(name='lrelu4')(x)
    # Block 5
    x = Conv2D(filters=128,
               kernel_size=5,
               strides=2,
               padding='same',
               name='conv5')(x)
    if use_elu:
        x = ELU(name='elu5')(x)
    else:
        x = LeakyReLU(name='lrelu5')(x)
    # Block 6
    x = Conv2D(filters=256,
               kernel_size=5,
               strides=1,
               padding='same',
               name='conv6')(x)
    if use_elu:
        x = ELU(name='elu6')(x)
    else:
        x = LeakyReLU(name='lrelu6')(x)
    # Block 7
    x = Conv2D(filters=256,
               kernel_size=5,
               strides=2,
               padding='same',
               name='conv7')(x)
    if use_elu:
        x = ELU(name='elu7')(x)
    else:
        x = LeakyReLU(name='lrelu7')(x)
    # Block 8
    x = Conv2D(filters=512,
               kernel_size=5,
               strides=1,
               padding='same',
               name='conv8')(x)
    if use_elu:
        x = ELU(name='elu8')(x)
    else:
        x = LeakyReLU(name='lrelu8')(x)
    # Block 9
    x = Conv2D(filters=512,
               kernel_size=4,
               strides=2,
               padding='same',
               name='conv9')(x)
    if use_elu:
        x = ELU(name='elu9')(x)
    else:
        x = LeakyReLU(name='lrelu9')(x)
    # Dense
    out_layer = Dense(1, name='dense', activation='sigmoid')(x)

    # Final model
    discriminator = Model(inputs=in_layer,
                          outputs=out_layer,
                          name='Discriminator')
    return discriminator


class DeepDeblur(Model):
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 use_elu: Optional[bool] = False,
                 num_res_blocks: Optional[int] = 19):
        super(DeepDeblur, self).__init__()

        # Build generator
        self.generator = create_generator(input_shape,
                                          use_elu,
                                          num_res_blocks)
        # Build discriminator
        self.discriminator = create_discriminator(input_shape,
                                                  use_elu)

        # Define and set loss functions
        K = 3

        # As content loss, a multiscale version of LogCosh is chosen
        def content_loss(sharp_pyramid: List[tf.Tensor],
                         predicted_pyramid: List[tf.Tensor]):
            # Check input
            assert len(sharp_pyramid) == K, 'The list \'trueY\' should contain {:d} elements'.format(K)
            assert len(predicted_pyramid) == K, 'The list \'predY\' should contain {:d} elements'.format(K)

            loss = 0.
            for scale_trueY, scale_predY in zip(sharp_pyramid, predicted_pyramid):
                scale_shape = scale_trueY.shape[1:]
                norm_factor = functools.reduce(operator.mul, scale_shape, 1)
                scale_loss = tf.reduce_sum(logcosh(scale_trueY, scale_predY)) / norm_factor
                loss += scale_loss
            return 1./(2. * K) * loss

        def total_loss(blurred_pyramid: List[tf.Tensor],
                       sharp_pyramid: List[tf.Tensor]):
            # Check input
            assert len(blurred_pyramid) == K, 'The list \'trueY\' should contain {:d} elements'.format(K)
            assert len(blurred_pyramid) == K, 'The list \'predY\' should contain {:d} elements'.format(K)

            predicted_pyramid = self.generator(blurred_pyramid)
            real_response = self.discriminator(sharp_pyramid)
            fake_response = self.discriminator(predicted_pyramid)
            adv_loss = tf.reduce_mean(tf.math.log(real_response) + tf.math.log(1 - fake_response))
            total = content_loss(sharp_pyramid, predicted_pyramid) + 1e-4 * adv_loss
            return total

        self.g_loss = total_loss
        self.d_loss = lambda real_response, fake_response: - tf.reduce_mean(tf.math.log(real_response) +
                                                                            tf.math.log(1 - fake_response))

        # Set optimizers as Adam with lr=1e-4
        self.g_optimizer = Adam(lr=1e-4)
        self.d_optimizer = Adam(lr=1e-4)

    @tf.function
    def train_step(self,
                   train_batch: Tuple[tf.Tensor, tf.Tensor]):
        # Determine height and width
        height = train_batch[0].shape[1]
        width = train_batch[0].shape[2]
        # Prepare Gaussian pyramid
        blurred_batch1 = train_batch[0]
        sharp_batch1 = train_batch[1]
        blurred_batch2 = tf.image.resize(train_batch[0], size=(height // 2, width // 2))
        sharp_batch2 = tf.image.resize(train_batch[1], size=(height // 2, width // 2))
        blurred_batch3 = tf.image.resize(train_batch[0], size=(height // 4, width // 4))
        sharp_batch3 = tf.image.resize(train_batch[1], size=(height // 4, width // 4))
        blurred_pyramid = [blurred_batch1, blurred_batch2, blurred_batch3]
        sharp_pyramid = [sharp_batch1, sharp_batch2, sharp_batch3]

        # Train the networks
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            # Make predictions
            predicted_pyramid = self.generator(blurred_pyramid, training=True)
            # Compute discriminator's output
            real_response = self.discriminator(sharp_pyramid, training=True)
            fake_response = self.discriminator(predicted_pyramid, training=True)
            # Calculate generator's and discriminator's loss
            g_loss = self.g_loss(sharp_pyramid,
                                 predicted_pyramid)
            d_loss = self.d_loss(real_response,
                                 fake_response)
        # Get gradient w.r.t. network's loss and update weights
        g_grad = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grad, self.generator.trainable_variables))
        d_grad = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))

        # Compute metrics
        ssim_metric = ssim(tf.cast(sharp_batch1, dtype='float32'),
                           predicted_pyramid[0])
        psnr_metric = psnr(tf.cast(sharp_batch1, dtype='float32'),
                           predicted_pyramid[0])

        return {'g_loss': g_loss,
                'd_loss': d_loss,
                'ssim': tf.reduce_mean(ssim_metric),
                'psnr': tf.reduce_mean(psnr_metric)}

    @tf.function
    def distributed_train_step(self,
                               train_batch: tf.data.Dataset,
                               strategy: Optional[tf.distribute.Strategy] = None):
        per_replica_results = strategy.run(self.train_step, args=(train_batch,))
        reduced_g_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                         per_replica_results['g_loss'], axis=None)
        reduced_d_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                         per_replica_results['d_loss'], axis=None)
        reduced_ssim = strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                       per_replica_results['ssim'], axis=None)
        reduced_psnr = strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                       per_replica_results['psnr'], axis=None)
        return {'g_loss': reduced_g_loss,
                'd_loss': reduced_d_loss,
                'ssim': reduced_ssim,
                'psnr': reduced_psnr}

    @tf.function
    def eval_step(self,
                  val_batch: Tuple[tf.Tensor, tf.Tensor]):
        # Determine height and width
        height = val_batch[0].shape[1]
        width = val_batch[0].shape[2]
        # Prepare Gaussian pyramid
        blurred_batch1 = val_batch[0]
        sharp_batch1 = val_batch[1]
        blurred_batch2 = tf.image.resize(val_batch[0], size=(height // 2, width // 2))
        sharp_batch2 = tf.image.resize(val_batch[1], size=(height // 2, width // 2))
        blurred_batch3 = tf.image.resize(val_batch[0], size=(height // 4, width // 4))
        sharp_batch3 = tf.image.resize(val_batch[1], size=(height // 4, width // 4))
        blurred_pyramid = [blurred_batch1, blurred_batch2, blurred_batch3]
        sharp_pyramid = [sharp_batch1, sharp_batch2, sharp_batch3]

        # Generate fake inputs
        predicted_pyramid = self.generator(blurred_pyramid, training=False)
        # Get logits for both fake and real images
        fake_logits = self.discriminator(sharp_pyramid, training=False)
        real_logits = self.discriminator(predicted_pyramid, training=False)
        # Calculate discriminator's loss
        d_loss_fake = self.d_loss(-tf.ones_like(fake_logits), fake_logits)
        d_loss_real = self.d_loss(tf.ones_like(real_logits), real_logits)
        d_loss = 0.5 * tf.add(d_loss_fake, d_loss_real)
        # Calculate generator's loss
        g_loss = self.g_loss(sharp_pyramid, predicted_pyramid)

        # Compute metrics
        ssim_metric = ssim(tf.cast(sharp_batch1, dtype='float32'),
                           predicted_pyramid[0])
        psnr_metric = psnr(tf.cast(sharp_batch1, dtype='float32'),
                           predicted_pyramid[0])

        return {'val_g_loss': g_loss,
                'val_d_loss': d_loss,
                'val_ssim': tf.reduce_mean(ssim_metric),
                'val_psnr': tf.reduce_mean(psnr_metric)}

    def distributed_train(self,
                          train_data: tf.data.Dataset,
                          epochs: int,
                          steps_per_epoch: int,
                          strategy: tf.distribute.Strategy,
                          initial_epoch: Optional[int] = 1,
                          validation_data: Optional[tf.data.Dataset] = None,
                          validation_steps: Optional[int] = None,
                          checkpoint_dir: Optional[str] = None):
        for ep in notebook.tqdm(range(initial_epoch, epochs + 1)):
            print('=' * 50)
            print('Epoch {:d}/{:d}'.format(ep, epochs))

            # Set up lists that will contain losses and metrics for each epoch
            g_losses = []
            d_losses = []
            ssim_metrics = []
            psnr_metrics = []

            # Perform training
            for batch in notebook.tqdm(train_data, total=steps_per_epoch):
                # Perform train step
                step_result = self.distributed_train_step(batch, strategy)

                # Collect results
                g_losses.append(step_result['g_loss'])
                d_losses.append(step_result['d_loss'])
                ssim_metrics.append(step_result['ssim'])
                psnr_metrics.append(step_result['psnr'])

            # Display training results
            train_results = 'g_loss: {:.4f} - d_loss: {:.4f} ssim: {:.4f} - psnr: {:.4f}'.format(
                np.mean(g_losses), np.mean(d_losses), np.mean(ssim_metrics), np.mean(psnr_metrics)
            )
            print(train_results)

            # Perform validation if required
            if validation_data is not None and validation_steps is not None:
                val_g_losses = []
                val_d_losses = []
                val_ssim_metrics = []
                val_psnr_metrics = []
                for val_batch in notebook.tqdm(validation_data, total=validation_steps):
                    # Perform eval step
                    step_result = self.eval_step(tf.cast(val_batch, dtype='float32'))

                    # Collect results
                    val_g_losses.append(step_result['val_g_loss'])
                    val_d_losses.append(step_result['val_d_loss'])
                    val_ssim_metrics.append(step_result['val_ssim'])
                    val_psnr_metrics.append(step_result['val_psnr'])

                # Display validation results
                val_results = 'val_g_loss: {:.4f} - val_g_loss: {:.4f} - val_ssim: {:.4f} - val_psnr: {:.4f}'.format(
                    np.mean(val_g_losses), np.mean(val_d_losses), np.mean(val_ssim_metrics), np.mean(val_psnr_metrics)
                )
                print(val_results)

            # Save model every 15 epochs if required
            if checkpoint_dir is not None and ep % 15 == 0:
                print('Saving generator\'s model...', end='')
                self.generator.save_weights(
                    filepath=os.path.join(checkpoint_dir, 'ep:{:03d}-psnr:{:.4f}.h5').format(
                        ep, np.mean(psnr_metrics)
                    )
                )
                print(' OK')
                print('Saving critic\'s model...', end='')
                self.critic.save_weights(
                    filepath=os.path.join(checkpoint_dir, 'ep:{:03d}-d_loss:{:.4f}.h5').format(
                        ep, np.mean(d_losses)
                    )
                )
                print(' OK')
