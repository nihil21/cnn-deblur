import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.layers import Input, Layer, Conv2D, Conv2DTranspose, Add, ELU, BatchNormalization, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.losses import logcosh
from tensorflow.keras.optimizers import Adam
from utils.custom_metrics import ssim, psnr
from tqdm import notebook
from typing import Tuple, List, Optional


def res_block(in_layer: Layer,
              layer_id: str,
              filters: Optional[int] = 64,
              kernels: Optional[int] = 5,
              use_batchnorm: Optional[bool] = True):
    # Block 1
    x = Conv2D(filters=filters,
               kernel_size=kernels,
               padding='same',
               name='res_conv{:s}_1'.format(layer_id))(in_layer)
    if use_batchnorm:
        x = BatchNormalization(name='res_bn{:s}_1'.format(layer_id))(x)
    x = ELU(name='res_elu{:s}_1'.format(layer_id))(x)
    # Block 2
    x = Conv2D(filters=filters,
               kernel_size=kernels,
               padding='same',
               name='res_conv{:s}_2'.format(layer_id))(x)
    if use_batchnorm:
        x = BatchNormalization(name='res_bn{:s}_2'.format(layer_id))(x)
    # Skip connection
    x = Add(name='res_add{:s}'.format(layer_id))([x, in_layer])
    x = ELU(name='res_elu{:s}_2'.format(layer_id))(x)
    return x


class DeepDeblur:
    def __init__(self, input_shape: Tuple[int, int, int]):
        # Coarsest branch
        in_layer3 = Input(shape=(input_shape[0] // 4, input_shape[1] // 4, input_shape[2]),
                          name='in_layer3')
        conv3 = Conv2D(filters=64,
                       kernel_size=5,
                       padding='same',
                       name='conv3')(in_layer3)
        x = conv3
        for i in range(19):
            x = res_block(in_layer=x,
                          layer_id='3_{:d}'.format(i))
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
        for i in range(19):
            x = res_block(in_layer=x,
                          layer_id='2_{:d}'.format(i))
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
        for i in range(19):
            x = res_block(in_layer=x,
                          layer_id='1_{:d}'.format(i))
        out_layer1 = Conv2D(filters=3,
                            kernel_size=5,
                            padding='same',
                            name='out_layer1')(x)

        # Final model
        self.model = Model(inputs=[in_layer1, in_layer2, in_layer3], outputs=[out_layer1, out_layer2, out_layer3])

        # Define and set loss function
        K = 3

        def multiscale_logcosh(trueY: List[tf.Tensor], predY: List[tf.Tensor]):
            loss = 0
            for s in range(K):
                scale_shape = trueY[s].shape
                loss += logcosh(trueY[s], predY[s]) / (scale_shape[0] * scale_shape[1] * scale_shape[2])
            return 1./(2. * K) * loss

        self.g_loss = multiscale_logcosh

        # Set optimizer as Adam with lr=1e-4
        self.g_optimizer = Adam(lr=1e-4)

    @tf.function
    def train_step(self,
                   train_batch: Tuple[tf.Tensor, tf.Tensor]):
        # Determine height and width
        height = train_batch[0].shape[0]
        width = train_batch[0].shape[1]
        # Prepare Gaussian pyramid
        blurred_batch1 = train_batch[0]
        sharp_batch1 = train_batch[1]
        blurred_batch2 = tf.image.resize(train_batch[0], size=(height // 2, width // 2))
        sharp_batch2 = tf.image.resize(train_batch[1], size=(height // 2, width // 2))
        blurred_batch3 = tf.image.resize(train_batch[0], size=(height // 4, width // 4))
        sharp_batch3 = tf.image.resize(train_batch[1], size=(height // 4, width // 4))

        # Train the network
        with tf.GradientTape() as g_tape:
            # Calculate generator's loss
            g_loss = self.g_loss([blurred_batch1, blurred_batch2, blurred_batch3],
                                 [sharp_batch1, sharp_batch2, sharp_batch3])
        # Get gradient w.r.t. network's loss and update weights
        g_grad = g_tape.gradient(g_loss, self.model.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grad, self.model.trainable_variables))

        # Compute metrics
        ssim_metrics = []
        psnr_metrics = []
        ssim_metrics.append(ssim(sharp_batch1, tf.cast(blurred_batch1, dtype='bfloat16')))
        psnr_metrics.append(psnr(sharp_batch1, tf.cast(blurred_batch1, dtype='bfloat16')))
        ssim_metrics.append(ssim(sharp_batch2, tf.cast(blurred_batch2, dtype='bfloat16')))
        psnr_metrics.append(psnr(sharp_batch2, tf.cast(blurred_batch2, dtype='bfloat16')))
        ssim_metrics.append(ssim(sharp_batch3, tf.cast(blurred_batch3, dtype='bfloat16')))
        psnr_metrics.append(psnr(sharp_batch3, tf.cast(blurred_batch3, dtype='bfloat16')))

        return {"g_loss": g_loss,
                "ssim": tf.reduce_mean(ssim_metrics),
                "psnr": tf.reduce_mean(psnr_metrics)}

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
        return {'g_loss': reduced_g_loss,
                'ssim': reduced_ssim,
                'psnr': reduced_psnr}

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

            # Perform training
            for batch in notebook.tqdm(train_data, total=steps_per_epoch):
                # Perform train step
                step_result = self.distributed_train_step(batch, strategy)

                # Collect results
                g_losses.append(step_result['g_loss'])
                ssim_metrics.append(step_result['ssim'])
                psnr_metrics.append(step_result['psnr'])

            # Display training results
            train_results = 'g_loss: {:.4f} - ssim: {:.4f} - psnr: {:.4f}'.format(
                np.mean(g_losses), np.mean(ssim_metrics), np.mean(psnr_metrics)
            )
            print(train_results)

            """# Perform validation if required
            if validation_data is not None and validation_steps is not None:
                val_g_losses = []
                val_ssim_metrics = []
                val_psnr_metrics = []
                for val_batch in notebook.tqdm(validation_data, total=validation_steps):
                    # Perform eval step
                    step_result = self.eval_step(tf.cast(val_batch, dtype='float32'))

                    # Collect results
                    val_d_losses.append(step_result['val_d_loss'])
                    val_g_losses.append(step_result['val_g_loss'])
                    val_ssim_metrics.append(step_result['val_ssim'])
                    val_psnr_metrics.append(step_result['val_psnr'])

                # Display validation results
                val_results = 'val_d_loss: {:.4f} - val_g_loss: {:.4f} - val_ssim: {:.4f} - val_psnr: {:.4f}'.format(
                    np.mean(val_d_losses), np.mean(val_g_losses), np.mean(val_ssim_metrics), np.mean(val_psnr_metrics)
                )
                print(val_results)"""

            # Save model every 15 epochs if required
            if checkpoint_dir is not None and ep % 15 == 0:
                print('Saving model...', end='')
                self.model.save_weights(
                    filepath=os.path.join(checkpoint_dir, 'ep:{:03d}-psnr:{:.4f}.h5').format(
                        ep, np.mean(psnr_metrics)
                    )
                )
                print(' OK')
