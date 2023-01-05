import os
import typing

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm.auto import tqdm



def create_patchgan_critic(
        input_shape,
        use_elu: bool = False,
        use_sigmoid: bool = False,
        use_bn: bool = False
):
    in_layer = keras.layers.Input(input_shape)
    # Block 1
    x = keras.layers.Conv2D(
        filters=64,
        kernel_size=4,
        strides=2,
        padding='same',
        name='conv1'
    )(in_layer)
    if use_bn:
        x = keras.layers.BatchNormalization(name='bn1')(x)
    else:
        x = keras.layers.LayerNormalization(name='ln1')(x)
    if use_elu:
        x = keras.layers.ELU(name='elu1')(x)
    else:
        x = keras.layers.LeakyReLU(name='lrelu1')(x)
    # Block 2
    x = keras.layers.Conv2D(
        filters=128,
        kernel_size=4,
        strides=2,
        padding='same',
        name='conv2'
    )(x)
    if use_bn:
        x = keras.layers.BatchNormalization(name='bn2')(x)
    else:
        x = keras.layers.LayerNormalization(name='ln2')(x)
    if use_elu:
        x = keras.layers.ELU(name='elu2')(x)
    else:
        x = keras.layers.LeakyReLU(name='lrelu2')(x)
    # Block 3
    x = keras.layers.Conv2D(
        filters=256,
        kernel_size=4,
        strides=2,
        padding='same',
        name='conv3'
    )(x)
    if use_bn:
        x = keras.layers.BatchNormalization(name='bn3')(x)
    else:
        x = keras.layers.LayerNormalization(name='ln3')(x)
    if use_elu:
        x = keras.layers.ELU(name='elu3')(x)
    else:
        x = keras.layers.LeakyReLU(name='lrelu3')(x)
    # Block 4
    x = keras.layers.Conv2D(
        filters=512,
        kernel_size=4,
        strides=1,
        padding='same',
        name='conv4'
    )(x)
    if use_bn:
        x = keras.layers.BatchNormalization(name='bn4')(x)
    else:
        x = keras.layers.LayerNormalization(name='ln4')(x)
    if use_elu:
        x = keras.layers.ELU(name='elu4')(x)
    else:
        x = keras.layers.LeakyReLU(name='lrelu4')(x)
    # Block 5
    out_layer = keras.layers.Conv2D(
        filters=1,
        kernel_size=4,
        strides=1,
        padding='same',
        name='conv5'
    )(x)
    if use_sigmoid:
        out_layer = keras.layers.Activation('sigmoid')(x)

    return keras.models.Model(inputs=in_layer, outputs=out_layer, name='Critic')


class WGAN:
    def __init__(
            self,
            generator: keras.models.Model,
            critic: keras.models.Model,
            generator_loss: typing.Callable,
            critic_loss: typing.Callable,
            generator_optimizer: keras.optimizers.Optimizer,
            critic_optimizer: keras.optimizers.Optimizer
    ):
        # Set generator's and critic's models
        self.generator = generator
        self.critic = critic

        # Set losses and optimizers
        self.g_loss = generator_loss
        self.c_loss = critic_loss
        self.g_optimizer = generator_optimizer
        self.c_optimizer = critic_optimizer

        # Set critic_updates, i.e. the times the critic gets trained w.r.t. one training step of the generator
        self.critic_updates = 5
        # Set weight of gradient penalty
        self.gp_weight = 10.0

    @tf.function
    def gradient_penalty(
            self,
            batch_size,
            real_imgs,
            fake_imgs
    ):
        # Get interpolated pyramid
        alpha = tf.random.normal(
            shape=[batch_size, 1, 1, 1],
            mean=0.,
            stddev=1.
        )
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

    # Override train_step and test_step in order to account for pyramids instead of single-scale images
    @tf.function
    def train_step(
            self,
            train_batch: typing.Tuple[tf.Tensor, tf.Tensor]
    ):
        (blurred_batch, sharp_batch) = train_batch
        batch_size = tf.shape(blurred_batch)[0]
        c_losses = []
        # Train the critic multiple times according to critic_updates (by default, 5)
        for _ in range(self.critic_updates):
            with tf.GradientTape() as c_tape:
                # Make predictions
                predicted_batch = self.generator(blurred_batch, training=True)
                # Get logits for both fake and real images (only original scale)
                fake_logits = self.critic(predicted_batch, training=True)
                real_logits = self.critic(sharp_batch, training=True)
                # Calculate critic's loss
                c_loss = self.c_loss(real_logits, fake_logits)
                # Calculate gradient penalty
                gp = self.gradient_penalty(batch_size,
                                           real_imgs=tf.cast(sharp_batch, dtype='float32'),
                                           fake_imgs=predicted_batch)
                # Add gradient penalty to the loss
                c_loss += gp * self.gp_weight
            # Get gradient w.r.t. critic's loss and update weights
            c_grad = c_tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(c_grad, self.critic.trainable_variables))

            c_losses.append(c_loss)

        # Train the generator
        with tf.GradientTape() as g_tape:
            # Make predictions
            predicted_batch = self.generator(blurred_batch, training=True)
            # Get logits for fake images (only original scale)
            fake_logits = self.critic(predicted_batch, training=True)
            # Calculate generator's loss
            g_loss = self.g_loss(sharp_batch, predicted_batch, fake_logits)
        # Get gradient w.r.t. generator's loss and update weights
        g_grad = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grad, self.generator.trainable_variables))

        # Compute metrics
        ssim_metric = tf.image.ssim(
            sharp_batch,
            tf.cast(predicted_batch, dtype='bfloat16'),
            max_val=2.
        )
        psnr_metric = tf.image.psnr(
            sharp_batch,
            tf.cast(predicted_batch, dtype='bfloat16'),
            max_val=2.
        )

        return {
            'g_loss': g_loss,
            'ssim': tf.reduce_mean(ssim_metric),
            'psnr': tf.reduce_mean(psnr_metric),
            'c_loss': tf.reduce_mean(c_losses)
        }

    @tf.function
    def distributed_train_step(
            self,
            train_batch: tf.data.Dataset,
            strategy: typing.Optional[tf.distribute.Strategy] = None
    ):
        per_replica_results = strategy.run(self.train_step, args=(train_batch,))
        reduced_g_loss = strategy.reduce(
            tf.distribute.ReduceOp.MEAN,
            per_replica_results['g_loss'], axis=None
        )
        reduced_ssim = strategy.reduce(
            tf.distribute.ReduceOp.MEAN,
            per_replica_results['ssim'], axis=None
        )
        reduced_psnr = strategy.reduce(
            tf.distribute.ReduceOp.MEAN,
            per_replica_results['psnr'], axis=None
        )
        reduced_c_loss = strategy.reduce(
            tf.distribute.ReduceOp.MEAN,
            per_replica_results['c_loss'], axis=None
        )
        return {
            'g_loss': reduced_g_loss,
            'ssim': reduced_ssim,
            'psnr': reduced_psnr,
            'c_loss': reduced_c_loss
        }

    @tf.function
    def test_step(
            self,
            val_batch: typing.Tuple[tf.Tensor, tf.Tensor]
    ):
        (blurred_batch, sharp_batch) = val_batch

        # Generate fake inputs
        predicted_batch = self.generator(blurred_batch, training=False)
        # Get logits for both fake and real images
        fake_logits = self.critic(sharp_batch, training=False)
        real_logits = self.critic(predicted_batch, training=False)
        # Calculate critic's loss
        c_loss = self.c_loss(real_logits, fake_logits)
        # Calculate generator's loss
        g_loss = self.g_loss(sharp_batch, predicted_batch, fake_logits)

        # Compute metrics
        ssim_metric = tf.image.ssim(
            sharp_batch,
            tf.cast(predicted_batch, dtype='bfloat16'),
            max_val=2.
        )
        psnr_metric = tf.image.psnr(
            sharp_batch,
            tf.cast(predicted_batch, dtype='bfloat16'),
            max_val=2.
        )

        return {
            'g_loss': g_loss,
            'ssim': tf.reduce_mean(ssim_metric),
            'psnr': tf.reduce_mean(psnr_metric),
            'c_loss': c_loss
        }

    @tf.function
    def distributed_train_step(
            self,
            train_batch: tf.data.Dataset,
            strategy: typing.Optional[tf.distribute.Strategy] = None
    ):
        per_replica_results = strategy.run(self.train_step, args=(train_batch,))
        reduced_g_loss = strategy.reduce(
            tf.distribute.ReduceOp.MEAN,
            per_replica_results['g_loss'], axis=None
        )
        reduced_ssim = strategy.reduce(
            tf.distribute.ReduceOp.MEAN,
            per_replica_results['ssim'], axis=None
        )
        reduced_psnr = strategy.reduce(
            tf.distribute.ReduceOp.MEAN,
            per_replica_results['psnr'], axis=None
        )
        reduced_c_loss = strategy.reduce(
            tf.distribute.ReduceOp.MEAN,
            per_replica_results['c_loss'], axis=None
        )
        return {
            'g_loss': reduced_g_loss,
            'ssim': reduced_ssim,
            'psnr': reduced_psnr,
            'c_loss': reduced_c_loss
        }

    def fit(
            self,
            train_data: typing.Union[tf.data.Dataset, np.ndarray],
            epochs: int,
            steps_per_epoch: int,
            initial_epoch: int = 0,
            validation_data: typing.Optional[tf.data.Dataset] = None,
            validation_steps: typing.Optional[int] = None,
            checkpoint_dir: typing.Optional[str] = None,
            checkpoint_freq: int = 15
    ):
        if isinstance(train_data, tf.data.Dataset):
            return self.__fit_on_dataset(
                train_data,
                epochs,
                steps_per_epoch,
                initial_epoch,
                validation_data,
                validation_steps,
                checkpoint_dir,
                checkpoint_freq
            )
        elif isinstance(train_data, typing.Tuple):
            return self.__fit_on_tensor(
                train_data,
                epochs,
                steps_per_epoch,
                initial_epoch,
                validation_data,
                validation_steps,
                checkpoint_dir,
                checkpoint_freq
            )

    def __fit_on_dataset(
            self,
            train_data: tf.data.Dataset,
            epochs: int,
            steps_per_epoch: int,
            initial_epoch: int = 0,
            validation_data: typing.Optional[tf.data.Dataset] = None,
            validation_steps: typing.Optional[int] = None,
            checkpoint_dir: typing.Optional[str] = None,
            checkpoint_freq: int = 15
    ):
        # Set up lists that will contain training history
        g_loss_hist = []
        ssim_hist = []
        psnr_hist = []
        c_loss_hist = []
        val_g_loss_hist = []
        val_ssim_hist = []
        val_psnr_hist = []
        val_c_loss_hist = []
        for ep in tqdm(range(initial_epoch, epochs)):
            print('=' * 50)
            print('Epoch {:d}/{:d}'.format(ep + 1, epochs))

            # Set up lists that will contain losses and metrics for each epoch
            g_losses = []
            ssim_metrics = []
            psnr_metrics = []
            c_losses = []

            # Perform training
            for train_batch in tqdm(train_data, total=steps_per_epoch):
                # Perform train step
                step_result = self.train_step(train_batch)

                # Collect results
                g_losses.append(step_result['g_loss'])
                ssim_metrics.append(step_result['ssim'])
                psnr_metrics.append(step_result['psnr'])
                c_losses.append(step_result['c_loss'])

            # Compute mean losses and metrics
            g_loss_mean = np.mean(g_losses)
            ssim_mean = np.mean(ssim_metrics)
            psnr_mean = np.mean(psnr_metrics)
            c_loss_mean = np.mean(c_losses)

            # Display training results
            train_results = 'g_loss: {:.4f} - ssim: {:.4f} - psnr: {:.4f} - '.format(
                g_loss_mean, ssim_mean, psnr_mean
            )
            train_results += 'c_loss: {:.4f}'.format(
                c_loss_mean
            )
            print(train_results)

            # Save results in training history
            g_loss_hist.append(g_loss_mean)
            ssim_hist.append(ssim_mean)
            psnr_hist.append(psnr_mean)
            c_loss_hist.append(c_loss_mean)

            # Perform validation if required
            if validation_data is not None and validation_steps is not None:
                val_g_losses = []
                val_ssim_metrics = []
                val_psnr_metrics = []
                val_c_losses = []
                for val_batch in tqdm(validation_data, total=validation_steps):
                    # Perform eval step
                    step_result = self.test_step(val_batch)

                    # Collect results
                    val_g_losses.append(step_result['g_loss'])
                    val_ssim_metrics.append(step_result['ssim'])
                    val_psnr_metrics.append(step_result['psnr'])
                    val_c_losses.append(step_result['c_loss'])

                # Compute mean losses and metrics
                val_g_loss_mean = np.mean(val_g_losses)
                val_ssim_mean = np.mean(val_ssim_metrics)
                val_psnr_mean = np.mean(val_psnr_metrics)
                val_c_loss_mean = np.mean(val_c_losses)

                # Display validation results
                val_results = 'val_g_loss: {:.4f} - val_ssim: {:.4f} - val_psnr: {:.4f} - '.format(
                    val_g_loss_mean, val_ssim_mean, val_psnr_mean
                )
                val_results += 'val_c_loss: {:.4f}'.format(
                    val_c_loss_mean
                )
                print(val_results)

                # Save results in training history
                val_g_loss_hist.append(val_g_loss_mean)
                val_ssim_hist.append(val_ssim_mean)
                val_psnr_hist.append(val_psnr_mean)
                val_c_loss_hist.append(val_c_loss_mean)

            # Save model every 15 epochs if required
            if checkpoint_dir is not None and (ep + 1) % checkpoint_freq == 0:
                print('Saving generator\'s model...', end='')
                self.generator.save_weights(
                    filepath=os.path.join(checkpoint_dir, 'ep:{:03d}-ssim:{:.4f}-psnr:{:.4f}.h5').format(
                        ep + 1, ssim_mean, psnr_mean
                    )
                )
                print(' OK')
                print('Saving critic\'s model...', end='')
                self.critic.save_weights(
                    filepath=os.path.join(checkpoint_dir, 'ep:{:03d}.h5').format(
                        ep + 1
                    )
                )
                print(' OK')

        # Return history
        return {
            'g_loss': g_loss_hist,
            'ssim': ssim_hist,
            'psnr': psnr_hist,
            'c_loss': c_loss_hist,
            'val_g_loss': val_g_loss_hist,
            'val_ssim': val_ssim_hist,
            'val_psnr': val_psnr_hist,
            'val_c_loss': val_c_loss_hist
        }

    def __fit_on_tensor(
            self,
            train_data: typing.Tuple[np.ndarray, np.ndarray],
            epochs: int,
            steps_per_epoch: int,
            initial_epoch: int = 0,
            validation_data: typing.Optional[typing.Tuple[np.ndarray, np.ndarray]] = None,
            validation_steps: typing.Optional[int] = None,
            checkpoint_dir: typing.Optional[str] = None,
            checkpoint_freq: int = 15
    ):
        batch_size = train_data[0].shape[0]
        val_batch_size = validation_data[0].shape[0] \
            if validation_data is not None and validation_steps is not None \
            else None
        # Set up lists that will contain training history
        g_loss_hist = []
        ssim_hist = []
        psnr_hist = []
        c_loss_hist = []
        val_g_loss_hist = []
        val_ssim_hist = []
        val_psnr_hist = []
        val_c_loss_hist = []
        for ep in tqdm(range(initial_epoch, epochs)):
            print('=' * 50)
            print('Epoch {:d}/{:d}'.format(ep + 1, epochs))

            # Permute indexes
            permuted_indexes = np.random.permutation(batch_size)

            # Set up lists that will contain losses and metrics for each epoch
            g_losses = []
            ssim_metrics = []
            psnr_metrics = []
            c_losses = []

            # Perform training
            for i in tqdm(range(steps_per_epoch)):
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

            # Compute mean losses and metrics
            g_loss_mean = np.mean(g_losses)
            ssim_mean = np.mean(ssim_metrics)
            psnr_mean = np.mean(psnr_metrics)
            c_loss_mean = np.mean(c_losses)

            # Display training results
            train_results = 'g_loss: {:.4f} - ssim: {:.4f} - psnr: {:.4f} - '.format(
                g_loss_mean, ssim_mean, psnr_mean
            )
            train_results += 'c_loss: {:.4f}'.format(
                c_loss_mean
            )
            print(train_results)

            # Save results in training history
            g_loss_hist.append(g_loss_mean)
            ssim_hist.append(ssim_mean)
            psnr_hist.append(psnr_mean)
            c_loss_hist.append(c_loss_mean)

            # Perform validation if required
            if validation_data is not None and validation_steps is not None:
                # Permute indexes
                val_permuted_indexes = np.random.permutation(val_batch_size)

                val_g_losses = []
                val_ssim_metrics = []
                val_psnr_metrics = []
                val_c_losses = []
                for i in tqdm(range(validation_steps)):
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

                # Compute mean losses and metrics
                val_g_loss_mean = np.mean(val_g_losses)
                val_ssim_mean = np.mean(val_ssim_metrics)
                val_psnr_mean = np.mean(val_psnr_metrics)
                val_c_loss_mean = np.mean(val_c_losses)

                # Display validation results
                val_results = 'val_g_loss: {:.4f} - val_ssim: {:.4f} - val_psnr: {:.4f} - '.format(
                    val_g_loss_mean, val_ssim_mean, val_psnr_mean
                )
                val_results += 'val_c_loss: {:.4f}'.format(
                    val_c_loss_mean
                )
                print(val_results)

                # Save results in training history
                val_g_loss_hist.append(val_g_loss_mean)
                val_ssim_hist.append(val_ssim_mean)
                val_psnr_hist.append(val_psnr_mean)
                val_c_loss_hist.append(val_c_loss_mean)

            # Save model every 15 epochs if required
            if checkpoint_dir is not None and (ep + 1) % checkpoint_freq == 0:
                print('Saving generator\'s model...', end='')
                self.generator.save_weights(
                    filepath=os.path.join(checkpoint_dir, 'ep:{:03d}-ssim:{:.4f}-psnr:{:.4f}.h5').format(
                        ep + 1, ssim_mean, psnr_mean
                    )
                )
                print(' OK')
                print('Saving critic\'s model...', end='')
                self.critic.save_weights(
                    filepath=os.path.join(checkpoint_dir, 'ep:{:03d}.h5').format(
                        ep + 1
                    )
                )
                print(' OK')

        # Return history
        return {
            'g_loss': g_loss_hist,
            'ssim': ssim_hist,
            'psnr': psnr_hist,
            'c_loss': c_loss_hist,
            'val_g_loss': val_g_loss_hist,
            'val_ssim': val_ssim_hist,
            'val_psnr': val_psnr_hist,
            'val_c_loss': val_c_loss_hist
        }

    def distributed_fit(
            self,
            train_data: tf.data.Dataset,
            epochs: int,
            steps_per_epoch: int,
            strategy: tf.distribute.Strategy,
            initial_epoch: typing.Optional[int] = 0,
            validation_data: typing.Optional[tf.data.Dataset] = None,
            validation_steps: typing.Optional[int] = None,
            checkpoint_dir: typing.Optional[str] = None,
            checkpoint_freq: typing.Optional[int] = 15
    ):
        # Set up lists that will contain training history
        g_loss_hist = []
        ssim_hist = []
        psnr_hist = []
        c_loss_hist = []
        val_g_loss_hist = []
        val_ssim_hist = []
        val_psnr_hist = []
        val_c_loss_hist = []
        for ep in tqdm(range(initial_epoch, epochs)):
            print('=' * 50)
            print('Epoch {:d}/{:d}'.format(ep + 1, epochs))

            # Set up lists that will contain losses and metrics for each epoch
            g_losses = []
            ssim_metrics = []
            psnr_metrics = []
            c_losses = []

            # Perform training
            for batch in tqdm(train_data, total=steps_per_epoch):
                # Perform train step
                step_result = self.distributed_train_step(batch, strategy)

                # Collect results
                g_losses.append(step_result['g_loss'])
                ssim_metrics.append(step_result['ssim'])
                psnr_metrics.append(step_result['psnr'])
                c_losses.append(step_result['c_loss'])

            # Compute mean losses and metrics
            g_loss_mean = np.mean(g_losses)
            ssim_mean = np.mean(ssim_metrics)
            psnr_mean = np.mean(psnr_metrics)
            c_loss_mean = np.mean(c_losses)

            # Display training results
            train_results = 'g_loss: {:.4f} - ssim: {:.4f} - psnr: {:.4f} - '.format(
                g_loss_mean, ssim_mean, psnr_mean
            )
            train_results += 'c_loss: {:.4f}'.format(
                c_loss_mean
            )
            print(train_results)

            # Save results in training history
            g_loss_hist.append(g_loss_mean)
            ssim_hist.append(ssim_mean)
            psnr_hist.append(psnr_mean)
            c_loss_hist.append(c_loss_mean)

            # Perform validation if required
            if validation_data is not None and validation_steps is not None:
                val_g_losses = []
                val_ssim_metrics = []
                val_psnr_metrics = []
                val_c_losses = []
                for val_batch in tqdm(validation_data, total=validation_steps):
                    # Perform eval step
                    step_result = self.test_step(val_batch)

                    # Collect results
                    val_g_losses.append(step_result['g_loss'])
                    val_ssim_metrics.append(step_result['ssim'])
                    val_psnr_metrics.append(step_result['psnr'])
                    val_c_losses.append(step_result['c_loss'])

                # Compute mean losses and metrics
                val_g_loss_mean = np.mean(val_g_losses)
                val_ssim_mean = np.mean(val_ssim_metrics)
                val_psnr_mean = np.mean(val_psnr_metrics)
                val_c_loss_mean = np.mean(val_c_losses)

                # Display validation results
                val_results = 'val_g_loss: {:.4f} - val_ssim: {:.4f} - val_psnr: {:.4f} - '.format(
                    val_g_loss_mean, val_ssim_mean, val_psnr_mean
                )
                val_results += 'val_c_loss: {:.4f}'.format(
                    val_c_loss_mean
                )
                print(val_results)

                # Save results in training history
                val_g_loss_hist.append(val_g_loss_mean)
                val_ssim_hist.append(val_ssim_mean)
                val_psnr_hist.append(val_psnr_mean)
                val_c_loss_hist.append(val_c_loss_mean)

            # Save model every 15 epochs if required
            if checkpoint_dir is not None and (ep + 1) % checkpoint_freq == 0:
                print('Saving generator\'s model...', end='')
                self.generator.save_weights(
                    filepath=os.path.join(checkpoint_dir, 'ep:{:03d}-ssim:{:.4f}-psnr:{:.4f}.h5').format(
                        ep + 1, ssim_mean, psnr_mean
                    )
                )
                print(' OK')
                print('Saving critic\'s model...', end='')
                self.critic.save_weights(
                    filepath=os.path.join(checkpoint_dir, 'ep:{:03d}.h5').format(
                        ep + 1
                    )
                )
                print(' OK')

        # Return history
        return {
            'g_loss': g_loss_hist,
            'ssim': ssim_hist,
            'psnr': psnr_hist,
            'c_loss': c_loss_hist,
            'val_g_loss': val_g_loss_hist,
            'val_ssim': val_ssim_hist,
            'val_psnr': val_psnr_hist,
            'val_c_loss': val_c_loss_hist
        }

    def evaluate(self,
                 test_data: tf.data.Dataset,
                 steps: int):
        g_losses = []
        ssim_metrics = []
        psnr_metrics = []
        c_losses = []
        for batch in tqdm(test_data, total=steps):
            # Perform test step
            step_result = self.test_step(batch)

            # Collect results
            g_losses.append(step_result['g_loss'])
            ssim_metrics.append(step_result['ssim'])
            psnr_metrics.append(step_result['psnr'])
            c_losses.append(step_result['c_loss'])

        # Compute mean losses and metrics
        g_loss_mean = np.mean(g_losses)
        ssim_mean = np.mean(ssim_metrics)
        psnr_mean = np.mean(psnr_metrics)
        c_loss_mean = np.mean(c_losses)

        # Display validation results
        results = 'g_loss: {:.4f}\nssim: {:.4f}\npsnr: {:.4f}\n'.format(
            g_loss_mean, ssim_mean, psnr_mean
        )
        results += 'c_loss: {:.4f}'.format(
            c_loss_mean
        )
        print(results)
