import numpy as np
from model.rednet import REDNet10
from model.custom_losses_metrics import psnr, ssim
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ELU, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import LogCosh
from tqdm import tqdm_notebook
from typing import Tuple, List, Optional


def create_generator(input_shape):
    generator = REDNet10(input_shape=input_shape).model
    generator._name = "Generator"
    return generator


def create_discriminator(input_shape,
                         filters: List[int],
                         kernels: List[int]):
    visible = Input(input_shape)

    x = visible
    for i in range(len(filters)):
        if i == 0:
            x = Conv2D(filters=filters[i],
                       kernel_size=kernels[i],
                       strides=2,
                       padding='same',
                       name='conv{:d}'.format(i))(x)
            x = BatchNormalization(name='bn{:d}'.format(i))(x)
        else:
            x = Conv2D(filters=filters[i],
                       kernel_size=kernels[i],
                       strides=1,
                       padding='same',
                       name='conv{:d}'.format(i))(x)
        x = ELU(name='act{:d}'.format(i))(x)

    x = Flatten(name='flat')(x)
    x = Dense(1024, activation='tanh', name='dense')(x)
    output = Dense(1, activation='sigmoid', name='output')(x)

    return Model(inputs=visible, outputs=output, name='Discriminator')


def create_combined(visible: Input,
                    generator: Model,
                    discriminator: Model):
    generated_images = generator(visible)
    outputs = discriminator(generated_images)
    return Model(inputs=visible, outputs=[generated_images, outputs])


class DeblurGan:
    def __init__(self, input_shape: Tuple[int, int, int]):
        # Define loss functions
        def wasserstein_loss(trueY, predY):
            return K.mean(trueY * predY)

        def perceptual_loss(trueY, predY):
            vgg = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
            loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
            loss_model.trainable = False
            for layer in loss_model.layers:
                layer.trainable = False
            return K.mean(K.square(loss_model(trueY) - loss_model(predY)))

        # Build generator
        self.generator = create_generator(input_shape)
        # Build and compile discriminator using Wasserstein loss
        self.discriminator = create_discriminator(input_shape,
                                                  filters=[64, 128, 256, 512],
                                                  kernels=[7, 3, 3, 3])
        self.discriminator.compile(Adam(lr=1e-4), loss=wasserstein_loss)
        # Build combined model
        visible = Input(input_shape)
        self.combined = create_combined(visible, self.generator, self.discriminator)

        # Compile combined model while freezing discriminator
        self.discriminator.trainable = False
        self.combined.compile(Adam(lr=1e-4),
                              loss=[LogCosh(), wasserstein_loss],
                              loss_weights=[100, 1])
        self.discriminator.trainable = True

    def train(self,
              train_data: tf.data.Dataset,
              batch_size: int,
              steps_per_epoch: int,
              epochs: int,
              critic_updates: Optional[int] = 5):
        output_true_batch = np.ones((batch_size, 1))
        output_false_batch = -np.ones((batch_size, 1))
        for ep in tqdm_notebook(range(epochs)):
            permuted_indexes = np.random.permutation(len(train_data[0]))
            d_losses = []
            c_losses = []
            psnr_metrics = []
            ssim_metrics = []
            for bat in tqdm_notebook(range(steps_per_epoch)):
                # Prepare batch
                batch_indexes = permuted_indexes[bat * batch_size:(bat + 1) * batch_size]
                blur_batch = train_data[0][batch_indexes]
                sharp_batch = train_data[1][batch_indexes]

                # blur_batch = None
                # sharp_batch = None
                # for batch in train_data.take(1):
                    # blur_batch = batch[0]
                    # sharp_batch = batch[1]

                # Generate fake inputs
                generated_batch = self.generator.predict(x=blur_batch, batch_size=batch_size)

                # Train discriminator
                for _ in range(critic_updates):
                    d_loss_real = self.discriminator.train_on_batch(sharp_batch, output_true_batch)
                    d_loss_fake = self.discriminator.train_on_batch(generated_batch, output_false_batch)
                    d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                    d_losses.append(d_loss)

                # Train generator only on discriminator's decisions
                self.discriminator.trainable = False
                c_loss = self.combined.train_on_batch(blur_batch, [sharp_batch, output_true_batch])
                c_losses.append(c_loss)
                psnr_metric = psnr(tf.convert_to_tensor(sharp_batch), tf.convert_to_tensor(generated_batch))
                psnr_metrics.append(psnr_metric)
                ssim_metric = ssim(tf.convert_to_tensor(sharp_batch), tf.convert_to_tensor(generated_batch))
                ssim_metrics.append(ssim_metric)
                self.discriminator.trainable = True

            # Display information for current epoch
            print('Ep: {:d} - DLoss: {:f} - CLoss: {:f} - PSNR: {:f} - SSIM: {:f}\n'
                  .format(ep,
                          np.mean(d_losses),
                          np.mean(c_losses),
                          np.mean(psnr_metrics),
                          np.mean(ssim_metrics)))
