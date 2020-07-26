import numpy as np
from model.rednet import REDNet10
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ELU, Flatten, Dense
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import Callback
import tqdm
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


def gan(visible: Input,
        generator: Model,
        discriminator: Model):
    generated_images = generator(visible)
    outputs = discriminator(generated_images)
    return Model(inputs=visible, outputs=[generated_images, outputs])


class DeblurGan:
    def __init__(self, input_shape: Tuple[int, int, int]):
        # Build GAN model
        self.generator = create_generator(input_shape)
        self.discriminator = create_discriminator(input_shape,
                                                  filters=[64, 128, 256, 512],
                                                  kernels=[7, 3, 3, 3])
        visible = Input(input_shape)
        self.model = gan(visible, self.generator, self.discriminator)

        # Define loss functions
        def perceptual_loss(trueY, predY):
            vgg = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
            loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
            loss_model.trainable = False
            return K.mean(K.square(loss_model(trueY) - loss_model(predY)))

        def wasserstein_loss(trueY, predY):
            return K.mean(trueY * predY)

        self.perceptual_loss = perceptual_loss
        self.wasserstein_loss = wasserstein_loss

    def compile(self):
        self.discriminator.trainable = True
        self.discriminator.compile(Adam(lr=1e-4),
                                   loss=self.wasserstein_loss)
        self.discriminator.trainable = False
        loss = [self.perceptual_loss, self.wasserstein_loss]
        loss_weights = [100, 1]
        self.model.compile(Adam(lr=1e-4),
                           loss=loss,
                           loss_weights=loss_weights)
        self.discriminator.trainable = True

    def train(self,
              train_data: tf.data.Dataset,
              batch_size: int,
              steps_per_epoch: int,
              epochs: int,
              critic_updates: Optional[int] = 5):
        output_true_batch = np.ones((batch_size, 1))
        output_false_batch = -np.ones((batch_size, 1))
        for ep in tqdm.tqdm(range(epochs)):
            permuted_indexes = np.random.permutation(len(train_data[0]))
            d_losses = []
            gan_losses = []
            for bat in range(steps_per_epoch):
                # Prepare batch
                batch_indexes = permuted_indexes[bat * batch_size:(bat + 1) * batch_size]
                blur_batch = train_data[0][batch_indexes]
                sharp_batch = train_data[1][batch_indexes]

                """blur_batch = None
                sharp_batch = None
                for batch in train_data.take(1):
                    blur_batch = batch[0]
                    sharp_batch = batch[1]"""

                # Generate fake inputs
                generated_images = self.generator.predict(x=blur_batch, batch_size=batch_size)

                # Train discriminator
                for _ in range(critic_updates):
                    d_loss_real = self.discriminator.train_on_batch(sharp_batch, output_true_batch)
                    d_loss_fake = self.discriminator.train_on_batch(generated_images, output_false_batch)
                    d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                    d_losses.append(d_loss)
                print("Ep. {:d}: discriminator trained".format(ep))

                self.discriminator.trainable = False
                # Train generator only on discriminator's decisions
                gan_loss = self.model.train_on_batch(blur_batch, [sharp_batch, output_true_batch])
                gan_losses.append(gan_loss)

                self.discriminator.trainable = True
