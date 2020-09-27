from models.conv_net import ConvNet
from models.wgan import WGAN, create_patchgan_critic
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Layer, Conv2D, Conv2DTranspose, Add, ELU, ReLU, Lambda,
                                     BatchNormalization, Activation, Reshape)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import logcosh, mse, mae
from typing import Tuple, List, Optional
from utils.custom_metrics import ssim, psnr
from tqdm import auto


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


class REDNetV2:
    def __init__(self, input_shape: Tuple[int, int, int],
                 num_layers: Optional[int] = 15,
                 learning_rate: Optional[float] = 1e-4):
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
                                name=f'{name}_dec_conv14')(x)
            x = BatchNormalization(name=f'{name}_dec_bn14')(x)
            x = ELU(name=f'{name}_dec_act14')(x)

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

        self.optimizer = Adam(lr=learning_rate)
        self.loss = logcosh

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
        mse_metric = mse(sharp_batch, merged)
        mae_metric = mae(sharp_batch, merged)
        return {'loss': loss_value,
                'ssim': ssim_metric,
                'psnr': psnr_metric,
                'mse': mse_metric,
                'mae': mae_metric}

    @tf.function
    def distributed_train_step(self,
                               train_batch: tf.data.Dataset,
                               strategy: Optional[tf.distribute.Strategy] = None):
        per_replica_results = strategy.run(self.train_step, args=(train_batch,))
        reduced_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                       per_replica_results['loss'], axis=None)
        reduced_ssim = strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                       per_replica_results['ssim'], axis=None)
        reduced_psnr = strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                       per_replica_results['psnr'], axis=None)
        reduced_mse = strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                      per_replica_results['mse'], axis=None)
        reduced_mae = strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                      per_replica_results['mae'], axis=None)
        return {'loss': reduced_loss,
                'ssim': reduced_ssim,
                'psnr': reduced_psnr,
                'mse': reduced_mse,
                'mae': reduced_mae}

    @tf.function
    def test_step(self,
                  val_batch: Tuple[tf.Tensor, tf.Tensor]):
        (blurred_batch, sharp_batch) = val_batch

        # Restore images and calculate loss
        sharp_channels = tf.split(sharp_batch, num_or_size_splits=3, axis=3)
        restored = self.model(blurred_batch, training=False)
        loss_red = self.loss(sharp_channels[0], restored[0])
        loss_green = self.loss(sharp_channels[1], restored[1])
        loss_blue = self.loss(sharp_channels[2], restored[2])
        loss_value = tf.reduce_mean(loss_red, loss_blue, loss_green)

        # Compute metrics
        merged = tf.concat(restored, axis=3)
        ssim_metric = ssim(sharp_batch, merged)
        psnr_metric = psnr(sharp_batch, merged)
        mse_metric = mse(sharp_batch, merged)
        mae_metric = mae(sharp_batch, merged)
        return {'loss': loss_value,
                'ssim': ssim_metric,
                'psnr': psnr_metric,
                'mse': mse_metric,
                'mae': mae_metric}

    def distributed_fit(self,
                        train_data: tf.data.Dataset,
                        epochs: int,
                        steps_per_epoch: int,
                        strategy: tf.distribute.Strategy,
                        initial_epoch: Optional[int] = 0,
                        validation_data: Optional[tf.data.Dataset] = None,
                        validation_steps: Optional[int] = None,
                        checkpoint_dir: Optional[str] = None,
                        checkpoint_freq: Optional[int] = 15):
        # Set up lists that will contain training history
        loss_hist = []
        ssim_hist = []
        psnr_hist = []
        mse_hist = []
        mae_hist = []
        val_loss_hist = []
        val_ssim_hist = []
        val_psnr_hist = []
        val_mse_hist = []
        val_mae_hist = []
        for ep in auto.tqdm(range(initial_epoch, epochs)):
            print('=' * 50)
            print('Epoch {:d}/{:d}'.format(ep + 1, epochs))

            # Set up lists that will contain losses and metrics for each epoch
            losses = []
            ssim_metrics = []
            psnr_metrics = []
            mse_metrics = []
            mae_metrics = []

            # Perform training
            for batch in auto.tqdm(train_data, total=steps_per_epoch):
                # Perform train step
                step_result = self.distributed_train_step(batch, strategy)

                # Collect results
                losses.append(step_result['loss'])
                ssim_metrics.append(step_result['ssim'])
                psnr_metrics.append(step_result['psnr'])
                mse_metrics.append(step_result['mse'])
                mae_metrics.append(step_result['mae'])

            # Compute mean losses and metrics
            loss_mean = np.mean(losses)
            ssim_mean = np.mean(ssim_metrics)
            psnr_mean = np.mean(psnr_metrics)
            mse_mean = np.mean(mse)
            mae_mean = np.mean(mae)

            # Display training results
            train_results = 'loss: {} - ssim: {:.4f} - psnr: {:.4f} - mse: {} - mae: {}'.format(
                loss_mean, ssim_mean, psnr_mean, mse_mean, mae_mean
            )
            print(train_results)

            # Save results in training history
            loss_hist.append(loss_mean)
            ssim_hist.append(ssim_mean)
            psnr_hist.append(psnr_mean)
            mse_hist.append(mse_mean)
            mae_hist.append(mae_mean)

            # Perform validation if required
            if validation_data is not None and validation_steps is not None:
                val_losses = []
                val_ssim_metrics = []
                val_psnr_metrics = []
                val_mse_metrics = []
                val_mae_metrics = []
                for val_batch in auto.tqdm(validation_data, total=validation_steps):
                    # Perform eval step
                    step_result = self.test_step(val_batch)

                    # Collect results
                    val_losses.append(step_result['loss'])
                    val_ssim_metrics.append(step_result['ssim'])
                    val_psnr_metrics.append(step_result['psnr'])
                    val_mse_metrics.append(step_result['mse'])
                    val_mae_metrics.append(step_result['mae'])

                # Compute mean losses and metrics
                val_loss_mean = np.mean(val_losses)
                val_ssim_mean = np.mean(val_ssim_metrics)
                val_psnr_mean = np.mean(val_psnr_metrics)
                val_mse_mean = np.mean(val_mse_metrics)
                val_mae_mean = np.mean(val_mae_metrics)

                # Display validation results
                val_results = 'val_loss: {} - val_ssim: {:.4f} - val_psnr: {:.4f} - val_mse: {} - val_mae : {}'.format(
                    val_loss_mean, val_ssim_mean, val_psnr_mean, val_mse_mean, val_mae_mean
                )
                print(val_results)

                # Save results in training history
                val_loss_hist.append(val_loss_mean)
                val_ssim_hist.append(val_ssim_mean)
                val_psnr_hist.append(val_psnr_mean)
                val_mse_hist.append(val_mse_mean)
                val_mae_hist.append(val_mae_mean)

            # Save model every 15 epochs if required
            if checkpoint_dir is not None and (ep + 1) % checkpoint_freq == 0:
                print('Saving model...', end='')
                self.model.save_weights(
                    filepath=os.path.join(checkpoint_dir, 'ep:{:03d}-ssim:{:.4f}-psnr:{:.4f}.h5').format(
                        ep + 1, ssim_mean, psnr_mean
                    )
                )
                print(' OK')

        # Return history
        return {'loss': loss_hist,
                'ssim': ssim_hist,
                'psnr': psnr_hist,
                'mse': mse_hist,
                'mae': mae_hist,
                'val_loss': val_loss_hist,
                'val_ssim': val_ssim_hist,
                'val_psnr': val_psnr_hist,
                'val_mse': val_mse_hist,
                'val_mae': val_mae_hist}

    def evaluate(self,
                 test_data: tf.data.Dataset,
                 steps: int):
        losses = []
        ssim_metrics = []
        psnr_metrics = []
        mse_metrics = []
        mae_metrics = []
        for batch in auto.tqdm(test_data, total=steps):
            # Perform test step
            step_result = self.test_step(batch)

            # Collect results
            losses.append(step_result['loss'])
            ssim_metrics.append(step_result['ssim'])
            psnr_metrics.append(step_result['psnr'])
            mse_metrics.append(step_result['mse'])
            mae_metrics.append(step_result['mae'])

        # Compute mean losses and metrics
        loss_mean = np.mean(losses)
        ssim_mean = np.mean(ssim_metrics)
        psnr_mean = np.mean(psnr_metrics)
        mse_mean = np.mean(mse_metrics)
        mae_mean = np.mean(mae_metrics)

        # Display validation results
        results = 'loss: {:.4f}\nssim: {:.4f}\npsnr: {:.4f}\nmse: {}\nmae: {}'.format(
            loss_mean, ssim_mean, psnr_mean, mse_mean, mae_mean
        )
        print(results)


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
