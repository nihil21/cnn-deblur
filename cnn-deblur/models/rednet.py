from models.conv_net import ConvNet
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Conv2D, Conv2DTranspose, Add, ELU, BatchNormalization, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from utils.custom_losses import ms_mse
from utils.custom_metrics import ssim, psnr
from tqdm import notebook
from typing import Tuple, List, Optional


def encode(in_layer: Layer,
           num_layers: Optional[int] = 15,
           filters: Optional[int] = 64,
           kernel_size: Optional[int] = 3,
           strides: Optional[int] = 1,
           padding: Optional[str] = 'same',
           scale_id: Optional[str] = '') -> List[Layer]:
    layers = []
    x = in_layer
    for i in range(num_layers):
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding=padding,
                   name='encode_conv{:d}{:s}'.format(i, scale_id))(x)
        x = ELU(name='encode_act{:d}{:s}'.format(i, scale_id))(x)
        x = BatchNormalization(name='encode_bn{:d}{:s}'.format(i, scale_id))(x)
        layers.append(x)
    return layers


def decode(res_layers: List[Layer],
           num_layers: Optional[int] = 15,
           filters: Optional[int] = 64,
           kernel_size: Optional[int] = 3,
           strides: Optional[int] = 1,
           padding: Optional[str] = 'same',
           scale_id: Optional[str] = '') -> List[Layer]:
    layers = []
    res_layers.reverse()
    x = res_layers[0]
    for i in range(num_layers):
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding=padding,
                   name='decode_conv{:d}{:s}'.format(i, scale_id))(x)
        if i % 2 != 0:
            x = Add(name='decode_skip{:d}{:s}'.format(i, scale_id))([x, res_layers[i]])
        x = ELU(name='decode_act{:d}{:s}'.format(i, scale_id))(x)
        x = BatchNormalization(name='decode_bn{:d}{:s}'.format(i, scale_id))(x)
        layers.append(x)

    return layers


"""class ConvBlock(Layer):
    def __init__(self,
                 name: str,
                 filters: int,
                 kernel_size: int = 3,
                 strides: int = 1,
                 padding: str = 'same',
                 activation: str = 'elu',
                 use_transpose: bool = False,
                 res_layer: Layer = None,
                 **kwargs):
        super().__init__(name, **kwargs)
        self.__conv_layer = Conv2D(filters=filters,
                                   kernel_size=kernel_size,
                                   strides=strides,
                                   padding=padding) if not use_transpose else Conv2DTranspose(filters=filters,
                                                                                              kernel_size=kernel_size,
                                                                                              strides=strides,
                                                                                              padding=padding)
        self.__res_layer = res_layer
        self.__activation = ELU() if activation == 'elu' else ReLU()
        self.__bn = BatchNormalization()

    def call(self, inputs, **kwargs):
        x = self.__conv_layer(inputs)
        if self.__res_layer is not None:
            x = Add()([x, self.__res_layer])
        x = self.__activation(x)
        return self.__bn(x)"""


class REDNet10(ConvNet):
    def __init__(self, input_shape: Tuple[int, int, int]):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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


class MSREDNet30:
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 learning_rate: Optional[int] = 1e-4):
        # --- Coarsest branch ---
        # ENCODER
        in_layer3 = Input(shape=(input_shape[0] // 4, input_shape[1] // 4, input_shape[2]),
                          name='in_layer3')
        encode_layers3 = encode(in_layer3, scale_id='3')
        # DECODER
        decode_layers3 = decode(encode_layers3, scale_id='3')
        out_layer3 = Conv2DTranspose(filters=3,
                                     kernel_size=1,
                                     strides=1,
                                     padding='same',
                                     name='output_conv3')(decode_layers3[-1])
        out_layer3 = Add(name='output_skip3')([out_layer3, in_layer3])
        out_layer3 = ELU(name='output_elu3')(out_layer3)
        # --- Middle branch ---
        # ENCODER
        in_layer2 = Input(shape=(input_shape[0] // 2, input_shape[1] // 2, input_shape[2]),
                          name='in_layer2')
        up_conv2 = Conv2DTranspose(filters=64,
                                   kernel_size=5,
                                   strides=2,
                                   padding='same')(out_layer3)
        concat2 = concatenate([in_layer2, up_conv2])
        encode_layers2 = encode(concat2, scale_id='2')
        # DECODER
        decode_layers2 = decode(encode_layers2, scale_id='2')
        out_layer2 = Conv2DTranspose(filters=3,
                                     kernel_size=1,
                                     strides=1,
                                     padding='same',
                                     name='output_conv2')(decode_layers2[-1])
        out_layer2 = Add(name='output_skip2')([out_layer2, in_layer2])
        out_layer2 = ELU(name='output_elu2')(out_layer2)
        # --- Finest branch ---
        # ENCODER
        in_layer1 = Input(shape=input_shape,
                          name='in_layer1')
        up_conv1 = Conv2DTranspose(filters=64,
                                   kernel_size=5,
                                   strides=2,
                                   padding='same')(out_layer2)
        concat1 = concatenate([in_layer1, up_conv1])
        encode_layers1 = encode(concat1, scale_id='1')
        # DECODER
        decode_layers1 = decode(encode_layers1, scale_id='1')
        out_layer1 = Conv2DTranspose(filters=3,
                                     kernel_size=1,
                                     strides=1,
                                     padding='same',
                                     name='output_conv1')(decode_layers1[-1])
        out_layer1 = Add(name='output_skip1')([out_layer1, in_layer1])
        out_layer1 = ELU(name='output_elu1')(out_layer1)

        # Build model
        self.model = Model(inputs=[in_layer1, in_layer2, in_layer3],
                           outputs=[out_layer1, out_layer2, out_layer3])

        # Set loss function and optimizer
        self.loss = ms_mse
        self.optimizer = Adam(lr=learning_rate)

    @tf.function
    def train_step(self,
                   train_batch: Tuple[tf.Tensor, tf.Tensor]):
        # Determine batch size, height and width
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

        # Train the generator
        with tf.GradientTape() as tape:
            # Make predictions
            predicted_pyramid = self.model(blurred_pyramid, training=True)
            # Calculate model's loss
            loss = self.loss(sharp_pyramid, predicted_pyramid)
        # Get gradient w.r.t. model's loss and update weights
        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

        # Compute metrics
        ssim_metric = ssim(sharp_pyramid[0],
                           predicted_pyramid[0])
        psnr_metric = psnr(sharp_pyramid[0],
                           predicted_pyramid[0])

        return {'loss': loss,
                'ssim': tf.reduce_mean(ssim_metric),
                'psnr': tf.reduce_mean(psnr_metric)}

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
        return {'loss': reduced_loss,
                'ssim': reduced_ssim,
                'psnr': reduced_psnr}

    @tf.function
    def test_step(self,
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

        # Make predictions
        predicted_pyramid = self.model(blurred_pyramid, training=False)
        # Calculate model's loss
        loss = self.loss(sharp_pyramid, predicted_pyramid)

        # Compute metrics
        ssim_metric = ssim(sharp_pyramid[0],
                           predicted_pyramid[0])
        psnr_metric = psnr(sharp_pyramid[0],
                           predicted_pyramid[0])

        return {'loss': loss,
                'ssim': tf.reduce_mean(ssim_metric),
                'psnr': tf.reduce_mean(psnr_metric)}

    def distributed_fit(self,
                        train_data: tf.data.Dataset,
                        epochs: int,
                        steps_per_epoch: int,
                        strategy: tf.distribute.Strategy,
                        initial_epoch: Optional[int] = 0,
                        validation_data: Optional[tf.data.Dataset] = None,
                        validation_steps: Optional[int] = None,
                        checkpoint_dir: Optional[str] = None):
        # Set up lists that will contain training history
        loss_hist = []
        ssim_hist = []
        psnr_hist = []
        val_loss_hist = []
        val_ssim_hist = []
        val_psnr_hist = []
        for ep in notebook.tqdm(range(initial_epoch, epochs)):
            print('=' * 50)
            print('Epoch {:d}/{:d}'.format(ep + 1, epochs))

            # Set up lists that will contain losses and metrics for each epoch
            losses = []
            ssim_metrics = []
            psnr_metrics = []

            # Perform training
            for batch in notebook.tqdm(train_data, total=steps_per_epoch):
                # Perform train step
                step_result = self.distributed_train_step(tf.cast(batch, dtype='float32'), strategy)

                # Collect results
                losses.append(step_result['loss'])
                ssim_metrics.append(step_result['ssim'])
                psnr_metrics.append(step_result['psnr'])

            # Compute mean losses and metrics
            loss_mean = np.mean(losses)
            ssim_mean = np.mean(ssim_metrics)
            psnr_mean = np.mean(psnr_metrics)

            # Display training results
            train_results = 'loss: {:.4f} - ssim: {:.4f} - psnr: {:.4f}'.format(
                loss_mean, ssim_mean, psnr_mean
            )
            print(train_results)

            # Save results in training history
            loss_hist.append(loss_mean)
            ssim_hist.append(ssim_mean)
            psnr_hist.append(psnr_mean)

            # Perform validation if required
            if validation_data is not None and validation_steps is not None:
                val_losses = []
                val_ssim_metrics = []
                val_psnr_metrics = []
                for val_batch in notebook.tqdm(validation_data, total=validation_steps):
                    # Perform eval step
                    step_result = self.test_step(tf.cast(val_batch, dtype='float32'))

                    # Collect results
                    val_losses.append(step_result['loss'])
                    val_ssim_metrics.append(step_result['ssim'])
                    val_psnr_metrics.append(step_result['psnr'])

                # Compute mean losses and metrics
                val_loss_mean = np.mean(val_losses)
                val_ssim_mean = np.mean(val_ssim_metrics)
                val_psnr_mean = np.mean(val_psnr_metrics)

                # Display validation results
                val_results = 'val_loss: {:.4f} - val_ssim: {:.4f} - val_psnr: {:.4f}'.format(
                    val_loss_mean, val_ssim_mean, val_psnr_mean
                )
                print(val_results)

                # Save results in training history
                val_loss_hist.append(val_loss_mean)
                val_ssim_hist.append(val_ssim_mean)
                val_psnr_hist.append(val_psnr_mean)

            # Save model every 15 epochs if required
            if checkpoint_dir is not None and (ep + 1) % 15 == 0:
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
                'val_loss': val_loss_hist,
                'val_ssim': val_ssim_hist,
                'val_psnr': val_psnr_hist}
