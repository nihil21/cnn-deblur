import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Layer, Conv2D, Conv2DTranspose, Add, Activation,
                                     ELU, ReLU, LeakyReLU, BatchNormalization, concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from utils.custom_losses import wasserstein_loss, content_loss
from tqdm import notebook
from typing import Tuple, List, Optional, Union


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


def create_critic(input_shape,
                  use_elu: Optional[bool] = False):
    in_layer = Input(input_shape)
    # Block 1
    x = Conv2D(filters=64,
               kernel_size=4,
               strides=2,
               padding='same',
               name='conv1')(in_layer)
    x = BatchNormalization(name='bn1')(x)
    if use_elu:
        x = ELU(name='elu1')(x)
    else:
        x = LeakyReLU(name='lrelu1')(x)
    # Block 2
    x = Conv2D(filters=128,
               kernel_size=4,
               strides=2,
               padding='same',
               name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    if use_elu:
        x = ELU(name='elu2')(x)
    else:
        x = LeakyReLU(name='lrelu2')(x)
    # Block 3
    x = Conv2D(filters=256,
               kernel_size=4,
               strides=2,
               padding='same',
               name='conv3')(x)
    x = BatchNormalization(name='bn3')(x)
    if use_elu:
        x = ELU(name='elu3')(x)
    else:
        x = LeakyReLU(name='lrelu3')(x)
    # Block 4
    x = Conv2D(filters=512,
               kernel_size=4,
               strides=1,
               padding='same',
               name='conv4')(x)
    x = BatchNormalization(name='bn4')(x)
    if use_elu:
        x = ELU(name='elu4')(x)
    else:
        x = LeakyReLU(name='lrelu4')(x)
    # Block 5
    x = Conv2D(filters=1,
               kernel_size=4,
               strides=1,
               padding='same',
               name='conv5')(x)
    out_layer = Activation('sigmoid')(x)

    return Model(inputs=in_layer, outputs=out_layer, name='Critic')


class DeepDeblurWGAN(Model):
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 use_elu: Optional[bool] = False,
                 num_res_blocks: Optional[int] = 19):
        super(DeepDeblurWGAN, self).__init__()

        # Build generator
        self.generator = create_generator(input_shape,
                                          use_elu,
                                          num_res_blocks)
        # Build critic
        self.critic = create_critic(input_shape,
                                    use_elu)

        # Define and set loss functions:
        # as content loss, a multiscale version of LogCosh is chosen
        def total_loss(blurred_pyramid: List[tf.Tensor],
                       sharp_pyramid: List[tf.Tensor]):
            # Check input
            assert len(blurred_pyramid) == 3, 'The list \'predY\' should contain {:d} elements'.format(3)
            assert len(sharp_pyramid) == 3, 'The list \'trueY\' should contain {:d} elements'.format(3)

            predicted_pyramid = self.generator(blurred_pyramid)
            fake_logits = self.critic(predicted_pyramid)
            adv_loss = tf.reduce_mean(-fake_logits)
            total = adv_loss + 100. * content_loss(sharp_pyramid, predicted_pyramid)
            return total

        self.g_loss = total_loss
        self.c_loss = wasserstein_loss

        # Set optimizers as Adam with lr=1e-4
        self.g_optimizer = Adam(lr=1e-4)
        self.c_optimizer = Adam(lr=1e-4)

        # Set critic_updates, i.e. the times the critic gets trained w.r.t. one training step of the generator
        self.critic_updates = 5
        # Set weight of gradient penalty
        self.gp_weight = 10.0

    @tf.function
    def gradient_penalty(self,
                         batch_size: int,
                         real_pyramid: List[tf.Tensor],
                         fake_pyramid: List[tf.Tensor]):
        # Get interpolated pyramid
        alpha = tf.random.normal(shape=[batch_size, 1, 1, 1],
                                 mean=0.0,
                                 stddev=1.0)
        diff = [fake_imgs - real_imgs
                for fake_imgs, real_imgs in zip(fake_pyramid, real_pyramid)]
        interpolated = [real_imgs + alpha * scale_diff
                        for real_imgs, scale_diff in zip(real_pyramid, diff)]

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # Get critic's output on the interpolated image
            pred = self.critic(interpolated, training=True)
        # Calculate gradients w.r.t. interpolated image
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # Calculate norm of gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        # Return gradient penalty
        return tf.reduce_mean((norm - 1.0) ** 2)

    @tf.function
    def train_step(self,
                   train_batch: Tuple[tf.Tensor, tf.Tensor]):
        # Determine batch size, height and width
        batch_size = train_batch[0].shape[0]
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

        c_losses = []
        # Train the critic multiple times according to critic_updates (by default, 5)
        for _ in range(self.critic_updates):
            with tf.GradientTape() as c_tape:
                # Make predictions
                predicted_pyramid = self.generator(blurred_pyramid, training=True)
                # Get logits for both fake and real images
                fake_logits = self.critic(predicted_pyramid, training=True)
                real_logits = self.critic(sharp_pyramid, training=True)
                # Calculate critic's loss
                c_loss_fake = self.c_loss(-tf.ones_like(fake_logits), fake_logits)
                c_loss_real = self.c_loss(tf.ones_like(real_logits), real_logits)
                c_loss = 0.5 * tf.add(c_loss_fake, c_loss_real)
                # Calculate gradient penalty
                gp = self.gradient_penalty(batch_size,
                                           real_pyramid=[tf.cast(sharp_imgs, dtype='float32')
                                                         for sharp_imgs in sharp_pyramid],
                                           fake_pyramid=predicted_pyramid)
                # Add gradient penalty to the loss
                c_loss += gp * self.gp_weight
            # Get gradient w.r.t. critic's loss and update weights
            c_grad = c_tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(c_grad, self.critic.trainable_variables))

            c_losses.append(c_loss)

        # Train the generator
        with tf.GradientTape() as g_tape:
            # Calculate generator's loss
            g_loss = self.g_loss(blurred_pyramid, sharp_pyramid)
        # Get gradient w.r.t. generator's loss and update weights
        g_grad = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grad, self.generator.trainable_variables))

        # Compute metrics
        ssim_metric = tf.image.ssim(sharp_pyramid[0],
                                    tf.cast(predicted_pyramid[0], dtype='bfloat16'),
                                    max_val=2.)
        psnr_metric = tf.image.psnr(sharp_pyramid[0],
                                    tf.cast(predicted_pyramid[0], dtype='bfloat16'),
                                    max_val=2.)
        real_l1_metric = tf.abs(tf.ones_like(real_logits) - real_logits)
        fake_l1_metric = tf.abs(-tf.ones_like(fake_logits) - fake_logits)

        return {'g_loss': g_loss,
                'ssim': tf.reduce_mean(ssim_metric),
                'psnr': tf.reduce_mean(psnr_metric),
                'c_loss': tf.reduce_mean(c_losses),
                'real_l1': tf.reduce_mean(real_l1_metric),
                'fake_l1': tf.reduce_mean(fake_l1_metric)}

    @tf.function
    def distributed_train_step(self,
                               train_batch: tf.data.Dataset,
                               strategy: Optional[tf.distribute.Strategy] = None):
        per_replica_results = strategy.run(self.train_step, args=(train_batch,))
        reduced_g_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                         per_replica_results['g_loss'], axis=None)
        reduced_ssim = strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                       per_replica_results['ssim'], axis=None)
        reduced_psnr = strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                       per_replica_results['psnr'], axis=None)
        reduced_c_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                         per_replica_results['c_loss'], axis=None)
        reduced_real_l1 = strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                          per_replica_results['real_l1'], axis=None)
        reduced_fake_l1 = strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                          per_replica_results['fake_l1'], axis=None)
        return {'g_loss': reduced_g_loss,
                'ssim': reduced_ssim,
                'psnr': reduced_psnr,
                'c_loss': reduced_c_loss,
                'real_l1': reduced_real_l1,
                'fake_l1': reduced_fake_l1}

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
        fake_logits = self.critic(sharp_pyramid, training=False)
        real_logits = self.critic(predicted_pyramid, training=False)
        # Calculate critic's loss
        c_loss_fake = self.c_loss(-tf.ones_like(fake_logits), fake_logits)
        c_loss_real = self.c_loss(tf.ones_like(real_logits), real_logits)
        c_loss = 0.5 * tf.add(c_loss_fake, c_loss_real)
        # Calculate generator's loss
        g_loss = self.g_loss(sharp_pyramid, predicted_pyramid)

        # Compute metrics
        ssim_metric = tf.image.ssim(sharp_pyramid[0],
                                    tf.cast(predicted_pyramid[0], dtype='bfloat16'),
                                    max_val=2.)
        psnr_metric = tf.image.psnr(sharp_pyramid[0],
                                    tf.cast(predicted_pyramid[0], dtype='bfloat16'),
                                    max_val=2.)
        real_l1_metric = tf.abs(tf.ones_like(real_logits) - real_logits)
        fake_l1_metric = tf.abs(-tf.ones_like(fake_logits) - fake_logits)

        return {'val_g_loss': g_loss,
                'val_ssim': tf.reduce_mean(ssim_metric),
                'val_psnr': tf.reduce_mean(psnr_metric),
                'val_c_loss': c_loss,
                'val_real_l1': tf.reduce_mean(real_l1_metric),
                'val_fake_l1': tf.reduce_mean(fake_l1_metric)}

    def train(self,
              train_data: Union[tf.data.Dataset, np.ndarray],
              epochs: int,
              steps_per_epoch: int,
              initial_epoch: Optional[int] = 1,
              validation_data: Optional[tf.data.Dataset] = None,
              validation_steps: Optional[int] = None,
              checkpoint_dir: Optional[str] = None):
        if isinstance(train_data, tf.data.Dataset):
            self.__train_on_dataset(train_data, epochs, steps_per_epoch,
                                    initial_epoch, validation_data, validation_steps, checkpoint_dir)
        elif isinstance(train_data, Tuple):
            self.__train_on_tensor(train_data, epochs, steps_per_epoch,
                                   initial_epoch, validation_data, validation_steps, checkpoint_dir)

    def __train_on_dataset(self,
                           train_data: tf.data.Dataset,
                           epochs: int,
                           steps_per_epoch: int,
                           initial_epoch: Optional[int] = 1,
                           validation_data: Optional[tf.data.Dataset] = None,
                           validation_steps: Optional[int] = None,
                           checkpoint_dir: Optional[str] = None):
        for ep in notebook.tqdm(range(initial_epoch, epochs + 1)):
            print('=' * 50)
            print('Epoch {:d}/{:d}'.format(ep, epochs))

            # Set up lists that will contain losses and metrics for each epoch
            g_losses = []
            ssim_metrics = []
            psnr_metrics = []
            c_losses = []
            real_l1_metrics = []
            fake_l1_metrics = []

            # Perform training
            for train_batch in notebook.tqdm(train_data, total=steps_per_epoch):
                # Perform train step
                step_result = self.train_step(train_batch)

                # Collect results
                g_losses.append(step_result['g_loss'])
                ssim_metrics.append(step_result['ssim'])
                psnr_metrics.append(step_result['psnr'])
                c_losses.append(step_result['c_loss'])
                real_l1_metrics.append(step_result['real_l1'])
                fake_l1_metrics.append(step_result['fake_l1'])

            # Display training results
            train_results = 'g_loss: {:.4f} - ssim: {:.4f} - psnr: {:.4f} - '.format(
                np.mean(g_losses), np.mean(ssim_metrics), np.mean(psnr_metrics)
            )
            train_results += 'c_loss: {:.4f} - real_l1: {:.4f} - fake_l1: {:.4f}'.format(
                np.mean(c_losses), np.mean(real_l1_metrics), np.mean(fake_l1_metrics)
            )
            print(train_results)

            # Perform validation if required
            if validation_data is not None and validation_steps is not None:
                val_g_losses = []
                val_ssim_metrics = []
                val_psnr_metrics = []
                val_c_losses = []
                val_real_l1_metrics = []
                val_fake_l1_metrics = []
                for val_batch in notebook.tqdm(validation_data, total=validation_steps):
                    # Perform eval step
                    step_result = self.eval_step(val_batch)

                    # Collect results
                    val_g_losses.append(step_result['val_g_loss'])
                    val_ssim_metrics.append(step_result['val_ssim'])
                    val_psnr_metrics.append(step_result['val_psnr'])
                    val_c_losses.append(step_result['val_c_loss'])
                    val_real_l1_metrics.append(step_result['val_real_l1'])
                    val_fake_l1_metrics.append(step_result['val_fake_l1'])

                # Display validation results
                val_results = 'val_g_loss: {:.4f} - val_ssim: {:.4f} - val_psnr: {:.4f} - '.format(
                    np.mean(val_g_losses), np.mean(val_ssim_metrics), np.mean(val_psnr_metrics),
                )
                val_results += 'val_c_loss: {:.4f} - val_real_l1: {:.4f} - val_fake_l1: {:.4f}'.format(
                    np.mean(val_c_losses), np.mean(val_real_l1_metrics), np.mean(val_fake_l1_metrics),
                )
                print(val_results)

            # Save model every 15 epochs if required
            if checkpoint_dir is not None and ep % 15 == 0:
                print('Saving generator\'s model...', end='')
                self.generator.save_weights(
                    filepath=os.path.join(checkpoint_dir, 'ep:{:03d}-ssim:{:.4f}-psnr:{:.4f}.h5').format(
                        ep, np.mean(ssim_metrics), np.mean(psnr_metrics)
                    )
                )
                print(' OK')
                print('Saving critic\'s model...', end='')
                self.critic.save_weights(
                    filepath=os.path.join(checkpoint_dir, 'ep:{:03d}-real_l1:{:.4f}-fake_l1:{:.4f}.h5').format(
                        ep, np.mean(real_l1_metrics), np.mean(fake_l1_metrics)
                    )
                )
                print(' OK')

    def __train_on_tensor(self,
                          train_data: Tuple[np.ndarray, np.ndarray],
                          epochs: int,
                          steps_per_epoch: int,
                          initial_epoch: Optional[int] = 1,
                          validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                          validation_steps: Optional[int] = None,
                          checkpoint_dir: Optional[str] = None):
        batch_size = train_data[0].shape[0]
        val_batch_size = validation_data[0].shape[0] \
            if validation_data is not None and validation_steps is not None \
            else None
        for ep in notebook.tqdm(range(initial_epoch, epochs + 1)):
            print('=' * 50)
            print('Epoch {:d}/{:d}'.format(ep, epochs))

            # Permute indexes
            permuted_indexes = np.random.permutation(batch_size)

            # Set up lists that will contain losses and metrics for each epoch
            g_losses = []
            ssim_metrics = []
            psnr_metrics = []
            c_losses = []
            real_l1_metrics = []
            fake_l1_metrics = []

            # Perform training
            for i in notebook.tqdm(range(steps_per_epoch)):
                # Prepare batch
                batch_indexes = permuted_indexes[i * batch_size:(i + 1) * batch_size]
                blurred_batch = train_data[0][batch_indexes]
                sharp_batch = train_data[1][batch_indexes]

                # Perform train step
                step_result = self.train_step((blurred_batch, sharp_batch))

                # Collect results
                g_losses.append(step_result['g_loss'])
                ssim_metrics.append(step_result['ssim'])
                psnr_metrics.append(step_result['psnr'])
                c_losses.append(step_result['c_loss'])
                real_l1_metrics.append(step_result['real_l1'])
                fake_l1_metrics.append(step_result['fake_l1'])

            # Display training results
            train_results = 'g_loss: {:.4f} - ssim: {:.4f} - psnr: {:.4f} - '.format(
                np.mean(g_losses), np.mean(ssim_metrics), np.mean(psnr_metrics)
            )
            train_results += 'c_loss: {:.4f} - real_l1: {:.4f} - fake_l1: {:.4f}'.format(
                np.mean(c_losses), np.mean(real_l1_metrics), np.mean(fake_l1_metrics)
            )
            print(train_results)

            # Perform validation if required
            if validation_data is not None and validation_steps is not None:
                # Permute indexes
                val_permuted_indexes = np.random.permutation(val_batch_size)

                val_g_losses = []
                val_ssim_metrics = []
                val_psnr_metrics = []
                val_c_losses = []
                val_real_l1_metrics = []
                val_fake_l1_metrics = []
                for i in notebook.tqdm(range(validation_steps)):
                    # Prepare batch
                    val_batch_indexes = val_permuted_indexes[i * val_batch_size:(i + 1) * val_batch_size]
                    val_blurred_batch = validation_data[0][val_batch_indexes]
                    val_sharp_batch = validation_data[1][val_batch_indexes]

                    # Perform eval step
                    step_result = self.eval_step((val_blurred_batch, val_sharp_batch))

                    # Collect results
                    val_g_losses.append(step_result['val_g_loss'])
                    val_ssim_metrics.append(step_result['val_ssim'])
                    val_psnr_metrics.append(step_result['val_psnr'])
                    val_c_losses.append(step_result['val_c_loss'])
                    val_real_l1_metrics.append(step_result['val_real_l1'])
                    val_fake_l1_metrics.append(step_result['val_fake_l1'])

                # Display validation results
                val_results = 'val_g_loss: {:.4f} - val_ssim: {:.4f} - val_psnr: {:.4f} - '.format(
                    np.mean(val_g_losses), np.mean(val_ssim_metrics), np.mean(val_psnr_metrics),
                )
                val_results += 'val_c_loss: {:.4f} - val_real_l1: {:.4f} - val_fake_l1: {:.4f}'.format(
                    np.mean(val_c_losses), np.mean(val_real_l1_metrics), np.mean(val_fake_l1_metrics),
                )
                print(val_results)

            # Save model every 15 epochs if required
            if checkpoint_dir is not None and ep % 15 == 0:
                print('Saving generator\'s model...', end='')
                self.generator.save_weights(
                    filepath=os.path.join(checkpoint_dir, 'ep:{:03d}-ssim:{:.4f}-psnr:{:.4f}.h5').format(
                        ep, np.mean(ssim_metrics), np.mean(psnr_metrics)
                    )
                )
                print(' OK')
                print('Saving critic\'s model...', end='')
                self.critic.save_weights(
                    filepath=os.path.join(checkpoint_dir, 'ep:{:03d}-real_l1:{:.4f}-fake_l1:{:.4f}.h5').format(
                        ep, np.mean(real_l1_metrics), np.mean(fake_l1_metrics)
                    )
                )
                print(' OK')

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
            ssim_metrics = []
            psnr_metrics = []
            c_losses = []
            real_l1_metrics = []
            fake_l1_metrics = []

            # Perform training
            for batch in notebook.tqdm(train_data, total=steps_per_epoch):
                # Perform train step
                step_result = self.distributed_train_step(batch, strategy)

                # Collect results
                g_losses.append(step_result['g_loss'])
                ssim_metrics.append(step_result['ssim'])
                psnr_metrics.append(step_result['psnr'])
                c_losses.append(step_result['c_loss'])
                real_l1_metrics.append(step_result['real_l1'])
                fake_l1_metrics.append(step_result['fake_l1'])

            # Display training results
            train_results = 'g_loss: {:.4f} - ssim: {:.4f} - psnr: {:.4f} - '.format(
                np.mean(g_losses), np.mean(ssim_metrics), np.mean(psnr_metrics)
            )
            train_results += 'c_loss: {:.4f} - real_l1: {:.4f} - fake_l1: {:.4f}'.format(
                np.mean(c_losses), np.mean(real_l1_metrics), np.mean(fake_l1_metrics)
            )
            print(train_results)

            # Perform validation if required
            if validation_data is not None and validation_steps is not None:
                val_g_losses = []
                val_ssim_metrics = []
                val_psnr_metrics = []
                val_c_losses = []
                val_real_l1_metrics = []
                val_fake_l1_metrics = []
                for val_batch in notebook.tqdm(validation_data, total=validation_steps):
                    # Perform eval step
                    step_result = self.eval_step(tf.cast(val_batch, dtype='float32'))

                    # Collect results
                    val_g_losses.append(step_result['val_g_loss'])
                    val_ssim_metrics.append(step_result['val_ssim'])
                    val_psnr_metrics.append(step_result['val_psnr'])
                    val_c_losses.append(step_result['val_c_loss'])
                    val_real_l1_metrics.append(step_result['val_real_l1'])
                    val_fake_l1_metrics.append(step_result['val_fake_l1'])

                # Display validation results
                val_results = 'val_g_loss: {:.4f} - val_ssim: {:.4f} - val_psnr: {:.4f} - '.format(
                    np.mean(val_g_losses), np.mean(val_ssim_metrics), np.mean(val_psnr_metrics),
                )
                val_results += 'val_c_loss: {:.4f} - val_real_l1: {:.4f} - val_fake_l1: {:.4f}'.format(
                    np.mean(val_c_losses), np.mean(val_real_l1_metrics), np.mean(val_fake_l1_metrics),
                )
                print(val_results)

            # Save model every 15 epochs if required
            if checkpoint_dir is not None and ep % 15 == 0:
                print('Saving generator\'s model...', end='')
                self.generator.save_weights(
                    filepath=os.path.join(checkpoint_dir, 'ep:{:03d}-ssim:{:.4f}-psnr:{:.4f}.h5').format(
                        ep, np.mean(ssim_metrics), np.mean(psnr_metrics)
                    )
                )
                print(' OK')
                print('Saving critic\'s model...', end='')
                self.critic.save_weights(
                    filepath=os.path.join(checkpoint_dir, 'ep:{:03d}-real_l1:{:.4f}-fake_l1:{:.4f}.h5').format(
                        ep, np.mean(real_l1_metrics), np.mean(fake_l1_metrics)
                    )
                )
                print(' OK')
