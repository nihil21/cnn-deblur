import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, ELU, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Optimizer
from utils.custom_metrics import ssim, psnr
from tqdm import notebook
from typing import Optional, Callable, Tuple, Union


def create_patchgan_critic(input_shape,
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


class WGAN:
    def __init__(self,
                 generator: Model,
                 critic: Model,
                 g_loss: Callable,
                 c_loss: Callable,
                 g_optimizer: Optimizer,
                 c_optimizer: Optimizer):
        # Set generator and critic
        self.generator = generator
        self.critic = critic

        # Set loss functions
        self.g_loss = g_loss
        self.c_loss = c_loss
        # Set optimizers
        self.g_optimizer = g_optimizer
        self.c_optimizer = c_optimizer

        # Set critic_updates, i.e. the times the critic gets trained w.r.t. one training step of the generator
        self.critic_updates = 5
        # Set weight of gradient penalty
        self.gp_weight = 10.0

    @tf.function
    def gradient_penalty(self,
                         batch_size,
                         real_imgs,
                         fake_imgs):
        # Get interpolated pyramid
        alpha = tf.random.normal(shape=[batch_size, 1, 1, 1],
                                 mean=0.,
                                 stddev=1.)
        interpolated = alpha * real_imgs + (1. - alpha) * fake_imgs

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
        blurred_batch = train_batch[0]
        sharp_batch = train_batch[1]
        batch_size = blurred_batch.shape[0]

        c_losses = []
        # Train the critic multiple times according to critic_updates (by default, 5)
        for _ in range(self.critic_updates):
            with tf.GradientTape() as c_tape:
                # Generate fake inputs
                generated_batch = self.generator(blurred_batch, training=True)
                # Get logits for both fake and real images
                fake_logits = self.critic(generated_batch, training=True)
                real_logits = self.critic(sharp_batch, training=True)
                # Calculate critic's loss
                c_loss = self.c_loss(real_logits, fake_logits)
                # Calculate gradient penalty
                gp = self.gradient_penalty(batch_size,
                                           real_imgs=tf.cast(sharp_batch, dtype='float32'),
                                           fake_imgs=generated_batch)
                # Add gradient penalty to the loss
                c_loss += gp * self.gp_weight
            # Get gradient w.r.t. critic's loss and update weights
            c_grad = c_tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(c_grad, self.critic.trainable_variables))

            c_losses.append(c_loss)

        # Train the generator
        with tf.GradientTape() as g_tape:
            # Generate fake inputs
            generated_batch = self.generator(blurred_batch, training=True)
            # Get logits for both fake and real images
            fake_logits = self.critic(generated_batch, training=True)
            # Calculate generator's loss
            g_loss = self.g_loss(sharp_batch, generated_batch, fake_logits)
        # Get gradient w.r.t. generator's loss and update weights
        g_grad = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grad, self.generator.trainable_variables))

        # Compute metrics
        ssim_metric = ssim(sharp_batch,
                           generated_batch)
        psnr_metric = psnr(sharp_batch,
                           generated_batch)
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
    def test_step(self,
                  val_batch: Tuple[tf.Tensor, tf.Tensor]):
        blurred_batch = val_batch[0]
        sharp_batch = val_batch[1]

        # Generate fake inputs
        generated_batch = self.generator(blurred_batch, training=False)
        # Get logits for both fake and real images
        fake_logits = self.critic(generated_batch, training=False)
        real_logits = self.critic(sharp_batch, training=False)
        # Calculate critic's loss
        c_loss = self.c_loss(real_logits, fake_logits)
        # Calculate generator's loss
        g_loss = self.g_loss(sharp_batch, generated_batch, fake_logits)

        # Compute metrics
        ssim_metric = ssim(sharp_batch,
                           generated_batch)
        psnr_metric = psnr(sharp_batch,
                           generated_batch)
        real_l1_metric = tf.abs(tf.ones_like(real_logits) - real_logits)
        fake_l1_metric = tf.abs(-tf.ones_like(fake_logits) - fake_logits)

        return {'g_loss': g_loss,
                'ssim': tf.reduce_mean(ssim_metric),
                'psnr': tf.reduce_mean(psnr_metric),
                'c_loss': c_loss,
                'real_l1': tf.reduce_mean(real_l1_metric),
                'fake_l1': tf.reduce_mean(fake_l1_metric)}

    def fit(self,
            train_data: Union[tf.data.Dataset, np.ndarray],
            epochs: int,
            steps_per_epoch: int,
            initial_epoch: Optional[int] = 1,
            validation_data: Optional[tf.data.Dataset] = None,
            validation_steps: Optional[int] = None,
            checkpoint_dir: Optional[str] = None):
        if isinstance(train_data, tf.data.Dataset):
            return self.__fit_on_dataset(train_data, epochs, steps_per_epoch,
                                         initial_epoch, validation_data, validation_steps, checkpoint_dir)
        elif isinstance(train_data, Tuple):
            return self.__fit_on_tensor(train_data, epochs, steps_per_epoch,
                                        initial_epoch, validation_data, validation_steps, checkpoint_dir)

    def __fit_on_dataset(self,
                         train_data: tf.data.Dataset,
                         epochs: int,
                         steps_per_epoch: int,
                         initial_epoch: Optional[int] = 1,
                         validation_data: Optional[tf.data.Dataset] = None,
                         validation_steps: Optional[int] = None,
                         checkpoint_dir: Optional[str] = None):
        # Set up lists that will contain training history
        g_loss_hist = []
        ssim_hist = []
        psnr_hist = []
        c_loss_hist = []
        real_l1_hist = []
        fake_l1_hist = []
        val_g_loss_hist = []
        val_ssim_hist = []
        val_psnr_hist = []
        val_c_loss_hist = []
        val_real_l1_hist = []
        val_fake_l1_hist = []
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
                step_result = self.train_step(tf.cast(train_batch, dtype='float32'))

                # Collect results
                g_losses.append(step_result['g_loss'])
                ssim_metrics.append(step_result['ssim'])
                psnr_metrics.append(step_result['psnr'])
                c_losses.append(step_result['c_loss'])
                real_l1_metrics.append(step_result['real_l1'])
                fake_l1_metrics.append(step_result['fake_l1'])

            # Compute mean losses and metrics
            g_loss_mean = np.mean(g_losses)
            ssim_mean = np.mean(ssim_metrics)
            psnr_mean = np.mean(psnr_metrics)
            c_loss_mean = np.mean(c_losses)
            real_l1_mean = np.mean(real_l1_metrics)
            fake_l1_mean = np.mean(fake_l1_metrics)

            # Display training results
            train_results = 'g_loss: {:.4f} - ssim: {:.4f} - psnr: {:.4f} - '.format(
                g_loss_mean, ssim_mean, psnr_mean
            )
            train_results += 'c_loss: {:.4f} - real_l1: {:.4f} - fake_l1: {:.4f}'.format(
                c_loss_mean, real_l1_mean, fake_l1_mean
            )
            print(train_results)

            # Save results in training history
            g_loss_hist.append(g_loss_mean)
            ssim_hist.append(ssim_mean)
            psnr_hist.append(psnr_mean)
            c_loss_hist.append(c_loss_mean)
            real_l1_hist.append(real_l1_mean)
            fake_l1_hist.append(fake_l1_mean)

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
                    step_result = self.test_step(tf.cast(val_batch, dtype='float32'))

                    # Collect results
                    val_g_losses.append(step_result['g_loss'])
                    val_ssim_metrics.append(step_result['ssim'])
                    val_psnr_metrics.append(step_result['psnr'])
                    val_c_losses.append(step_result['c_loss'])
                    val_real_l1_metrics.append(step_result['real_l1'])
                    val_fake_l1_metrics.append(step_result['fake_l1'])

                # Compute mean losses and metrics
                val_g_loss_mean = np.mean(val_g_losses)
                val_ssim_mean = np.mean(val_ssim_metrics)
                val_psnr_mean = np.mean(val_psnr_metrics)
                val_c_loss_mean = np.mean(val_c_losses)
                val_real_l1_mean = np.mean(val_real_l1_metrics)
                val_fake_l1_mean = np.mean(val_fake_l1_metrics)

                # Display validation results
                val_results = 'val_g_loss: {:.4f} - val_ssim: {:.4f} - val_psnr: {:.4f} - '.format(
                    val_g_loss_mean, val_ssim_mean, val_psnr_mean
                )
                val_results += 'val_c_loss: {:.4f} - val_real_l1: {:.4f} - val_fake_l1: {:.4f}'.format(
                    val_c_loss_mean, val_real_l1_mean, val_fake_l1_mean
                )
                print(val_results)

                # Save results in training history
                val_g_loss_hist.append(val_g_loss_mean)
                val_ssim_hist.append(val_ssim_mean)
                val_psnr_hist.append(val_psnr_mean)
                val_c_loss_hist.append(val_c_loss_mean)
                val_real_l1_hist.append(val_real_l1_mean)
                val_fake_l1_hist.append(val_fake_l1_mean)

            # Save model every 15 epochs if required
            if checkpoint_dir is not None and ep % 15 == 0:
                print('Saving generator\'s model...', end='')
                self.generator.save_weights(
                    filepath=os.path.join(checkpoint_dir, 'ep:{:03d}-ssim:{:.4f}-psnr:{:.4f}.h5').format(
                        ep, ssim_mean, psnr_mean
                    )
                )
                print(' OK')
                print('Saving critic\'s model...', end='')
                self.critic.save_weights(
                    filepath=os.path.join(checkpoint_dir, 'ep:{:03d}-real_l1:{:.4f}-fake_l1:{:.4f}.h5').format(
                        ep, real_l1_mean, fake_l1_mean
                    )
                )
                print(' OK')

        # Return history
        return {'g_loss': g_loss_hist,
                'ssim': ssim_hist,
                'psnr': psnr_hist,
                'c_loss': c_loss_hist,
                'real_l1': real_l1_hist,
                'fake_l1': fake_l1_hist,
                'val_g_loss': val_g_loss_hist,
                'val_ssim': val_ssim_hist,
                'val_psnr': val_psnr_hist,
                'val_c_loss': val_c_loss_hist,
                'val_real_l1': val_real_l1_hist,
                'val_fake_l1': val_fake_l1_hist}

    def __fit_on_tensor(self,
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
        # Set up lists that will contain training history
        g_loss_hist = []
        ssim_hist = []
        psnr_hist = []
        c_loss_hist = []
        real_l1_hist = []
        fake_l1_hist = []
        val_g_loss_hist = []
        val_ssim_hist = []
        val_psnr_hist = []
        val_c_loss_hist = []
        val_real_l1_hist = []
        val_fake_l1_hist = []
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

            # Compute mean losses and metrics
            g_loss_mean = np.mean(g_losses)
            ssim_mean = np.mean(ssim_metrics)
            psnr_mean = np.mean(psnr_metrics)
            c_loss_mean = np.mean(c_losses)
            real_l1_mean = np.mean(real_l1_metrics)
            fake_l1_mean = np.mean(fake_l1_metrics)

            # Display training results
            train_results = 'g_loss: {:.4f} - ssim: {:.4f} - psnr: {:.4f} - '.format(
                g_loss_mean, ssim_mean, psnr_mean
            )
            train_results += 'c_loss: {:.4f} - real_l1: {:.4f} - fake_l1: {:.4f}'.format(
                c_loss_mean, real_l1_mean, fake_l1_mean
            )
            print(train_results)

            # Save results in training history
            g_loss_hist.append(g_loss_mean)
            ssim_hist.append(ssim_mean)
            psnr_hist.append(psnr_mean)
            c_loss_hist.append(c_loss_mean)
            real_l1_hist.append(real_l1_mean)
            fake_l1_hist.append(fake_l1_mean)

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
                    step_result = self.test_step((val_blurred_batch, val_sharp_batch))

                    # Collect results
                    val_g_losses.append(step_result['g_loss'])
                    val_ssim_metrics.append(step_result['ssim'])
                    val_psnr_metrics.append(step_result['psnr'])
                    val_c_losses.append(step_result['c_loss'])
                    val_real_l1_metrics.append(step_result['real_l1'])
                    val_fake_l1_metrics.append(step_result['fake_l1'])

                # Compute mean losses and metrics
                val_g_loss_mean = np.mean(val_g_losses)
                val_ssim_mean = np.mean(val_ssim_metrics)
                val_psnr_mean = np.mean(val_psnr_metrics)
                val_c_loss_mean = np.mean(val_c_losses)
                val_real_l1_mean = np.mean(val_real_l1_metrics)
                val_fake_l1_mean = np.mean(val_fake_l1_metrics)

                # Display validation results
                val_results = 'val_g_loss: {:.4f} - val_ssim: {:.4f} - val_psnr: {:.4f} - '.format(
                    val_g_loss_mean, val_ssim_mean, val_psnr_mean
                )
                val_results += 'val_c_loss: {:.4f} - val_real_l1: {:.4f} - val_fake_l1: {:.4f}'.format(
                    val_c_loss_mean, val_real_l1_mean, val_fake_l1_mean
                )
                print(val_results)

                # Save results in training history
                val_g_loss_hist.append(val_g_loss_mean)
                val_ssim_hist.append(val_ssim_mean)
                val_psnr_hist.append(val_psnr_mean)
                val_c_loss_hist.append(val_c_loss_mean)
                val_real_l1_hist.append(val_real_l1_mean)
                val_fake_l1_hist.append(val_fake_l1_mean)

            # Save model every 15 epochs if required
            if checkpoint_dir is not None and ep % 15 == 0:
                print('Saving generator\'s model...', end='')
                self.generator.save_weights(
                    filepath=os.path.join(checkpoint_dir, 'ep:{:03d}-ssim:{:.4f}-psnr:{:.4f}.h5').format(
                        ep, ssim_mean, psnr_mean
                    )
                )
                print(' OK')
                print('Saving critic\'s model...', end='')
                self.critic.save_weights(
                    filepath=os.path.join(checkpoint_dir, 'ep:{:03d}-real_l1:{:.4f}-fake_l1:{:.4f}.h5').format(
                        ep, real_l1_mean, fake_l1_mean
                    )
                )
                print(' OK')

        # Return history
        return {'g_loss': g_loss_hist,
                'ssim': ssim_hist,
                'psnr': psnr_hist,
                'c_loss': c_loss_hist,
                'real_l1': real_l1_hist,
                'fake_l1': fake_l1_hist,
                'val_g_loss': val_g_loss_hist,
                'val_ssim': val_ssim_hist,
                'val_psnr': val_psnr_hist,
                'val_c_loss': val_c_loss_hist,
                'val_real_l1': val_real_l1_hist,
                'val_fake_l1': val_fake_l1_hist}

    def distributed_fit(self,
                        train_data: tf.data.Dataset,
                        epochs: int,
                        steps_per_epoch: int,
                        strategy: tf.distribute.Strategy,
                        initial_epoch: Optional[int] = 1,
                        validation_data: Optional[tf.data.Dataset] = None,
                        validation_steps: Optional[int] = None,
                        checkpoint_dir: Optional[str] = None):
        # Set up lists that will contain training history
        g_loss_hist = []
        ssim_hist = []
        psnr_hist = []
        c_loss_hist = []
        real_l1_hist = []
        fake_l1_hist = []
        val_g_loss_hist = []
        val_ssim_hist = []
        val_psnr_hist = []
        val_c_loss_hist = []
        val_real_l1_hist = []
        val_fake_l1_hist = []
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
                step_result = self.distributed_train_step(tf.cast(batch, dtype='float32'), strategy)

                # Collect results
                g_losses.append(step_result['g_loss'])
                ssim_metrics.append(step_result['ssim'])
                psnr_metrics.append(step_result['psnr'])
                c_losses.append(step_result['c_loss'])
                real_l1_metrics.append(step_result['real_l1'])
                fake_l1_metrics.append(step_result['fake_l1'])

            # Compute mean losses and metrics
            g_loss_mean = np.mean(g_losses)
            ssim_mean = np.mean(ssim_metrics)
            psnr_mean = np.mean(psnr_metrics)
            c_loss_mean = np.mean(c_losses)
            real_l1_mean = np.mean(real_l1_metrics)
            fake_l1_mean = np.mean(fake_l1_metrics)

            # Display training results
            train_results = 'g_loss: {:.4f} - ssim: {:.4f} - psnr: {:.4f} - '.format(
                g_loss_mean, ssim_mean, psnr_mean
            )
            train_results += 'c_loss: {:.4f} - real_l1: {:.4f} - fake_l1: {:.4f}'.format(
                c_loss_mean, real_l1_mean, fake_l1_mean
            )
            print(train_results)

            # Save results in training history
            g_loss_hist.append(g_loss_mean)
            ssim_hist.append(ssim_mean)
            psnr_hist.append(psnr_mean)
            c_loss_hist.append(c_loss_mean)
            real_l1_hist.append(real_l1_mean)
            fake_l1_hist.append(fake_l1_mean)

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
                    step_result = self.test_step(tf.cast(val_batch, dtype='float32'))

                    # Collect results
                    val_g_losses.append(step_result['g_loss'])
                    val_ssim_metrics.append(step_result['ssim'])
                    val_psnr_metrics.append(step_result['psnr'])
                    val_c_losses.append(step_result['c_loss'])
                    val_real_l1_metrics.append(step_result['real_l1'])
                    val_fake_l1_metrics.append(step_result['fake_l1'])

                # Compute mean losses and metrics
                val_g_loss_mean = np.mean(val_g_losses)
                val_ssim_mean = np.mean(val_ssim_metrics)
                val_psnr_mean = np.mean(val_psnr_metrics)
                val_c_loss_mean = np.mean(val_c_losses)
                val_real_l1_mean = np.mean(val_real_l1_metrics)
                val_fake_l1_mean = np.mean(val_fake_l1_metrics)

                # Display validation results
                val_results = 'val_g_loss: {:.4f} - val_ssim: {:.4f} - val_psnr: {:.4f} - '.format(
                    val_g_loss_mean, val_ssim_mean, val_psnr_mean
                )
                val_results += 'val_c_loss: {:.4f} - val_real_l1: {:.4f} - val_fake_l1: {:.4f}'.format(
                    val_c_loss_mean, val_real_l1_mean, val_fake_l1_mean
                )
                print(val_results)

                # Save results in training history
                val_g_loss_hist.append(val_g_loss_mean)
                val_ssim_hist.append(val_ssim_mean)
                val_psnr_hist.append(val_psnr_mean)
                val_c_loss_hist.append(val_c_loss_mean)
                val_real_l1_hist.append(val_real_l1_mean)
                val_fake_l1_hist.append(val_fake_l1_mean)

            # Save model every 15 epochs if required
            if checkpoint_dir is not None and ep % 15 == 0:
                print('Saving generator\'s model...', end='')
                self.generator.save_weights(
                    filepath=os.path.join(checkpoint_dir, 'ep:{:03d}-ssim:{:.4f}-psnr:{:.4f}.h5').format(
                        ep, ssim_mean, psnr_mean
                    )
                )
                print(' OK')
                print('Saving critic\'s model...', end='')
                self.critic.save_weights(
                    filepath=os.path.join(checkpoint_dir, 'ep:{:03d}-real_l1:{:.4f}-fake_l1:{:.4f}.h5').format(
                        ep, real_l1_mean, fake_l1_mean
                    )
                )
                print(' OK')

        # Return history
        return {'g_loss': g_loss_hist,
                'ssim': ssim_hist,
                'psnr': psnr_hist,
                'c_loss': c_loss_hist,
                'real_l1': real_l1_hist,
                'fake_l1': fake_l1_hist,
                'val_g_loss': val_g_loss_hist,
                'val_ssim': val_ssim_hist,
                'val_psnr': val_psnr_hist,
                'val_c_loss': val_c_loss_hist,
                'val_real_l1': val_real_l1_hist,
                'val_fake_l1': val_fake_l1_hist}

    def evaluate(self,
                 test_data: tf.data.Dataset,
                 steps: int):
        g_losses = []
        ssim_metrics = []
        psnr_metrics = []
        c_losses = []
        real_l1_metrics = []
        fake_l1_metrics = []
        for batch in notebook.tqdm(test_data, total=steps):
            # Perform test step
            step_result = self.test_step(tf.cast(batch, dtype='float32'))

            # Collect results
            g_losses.append(step_result['g_loss'])
            ssim_metrics.append(step_result['ssim'])
            psnr_metrics.append(step_result['psnr'])
            c_losses.append(step_result['c_loss'])
            real_l1_metrics.append(step_result['real_l1'])
            fake_l1_metrics.append(step_result['fake_l1'])

        # Compute mean losses and metrics
        g_loss_mean = np.mean(g_losses)
        ssim_mean = np.mean(ssim_metrics)
        psnr_mean = np.mean(psnr_metrics)
        c_loss_mean = np.mean(c_losses)
        real_l1_mean = np.mean(real_l1_metrics)
        fake_l1_mean = np.mean(fake_l1_metrics)

        # Display validation results
        results = 'g_loss: {:.4f}\nssim: {:.4f}\npsnr: {:.4f}\n'.format(
            g_loss_mean, ssim_mean, psnr_mean
        )
        results += 'c_loss: {:.4f}\nreal_l1: {:.4f}\nfake_l1: {:.4f}'.format(
            c_loss_mean, real_l1_mean, fake_l1_mean
        )
        print(results)
