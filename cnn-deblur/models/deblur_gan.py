from models.rednet import REDNet10
from utils.custom_losses import wasserstein_loss, perceptual_loss
from utils.custom_metrics import psnr, ssim
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ELU, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from typing import Tuple, List

# Function to build generator and critic
def create_generator(input_shape):
    generator = REDNet10(input_shape=input_shape).model
    generator._name = "Generator"
    return generator


def create_critic(input_shape,
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


class DeblurGan(Model):
    def __init__(self, input_shape: Tuple[int, int, int]):
        super(DeblurGan, self).__init__()

        # Build generator
        self.generator = create_generator(input_shape)
        # Build critic (discriminator)
        self.critic = create_critic(input_shape,
                                    filters=[64, 128, 256, 512],
                                    kernels=[7, 3, 3, 3])
        # Define optimizers
        self.g_optimizer = Adam(lr=1e-4)
        self.d_optimizer = Adam(lr=1e-4)

        # Generator's loss function
        def total_loss(blurred_batch: tf.Tensor,
                       sharp_batch: tf.Tensor):
            generated_batch = self.generator(blurred_batch, reuse=False)
            fake_logits = self.critic(generated_batch, reuse=False)
            adv_loss = tf.reduce_mean(-fake_logits)
            content_loss = perceptual_loss(sharp_batch, generated_batch)
            return adv_loss + 100.0 * content_loss

        # Set loss functions
        self.d_loss = wasserstein_loss
        self.g_loss = total_loss

        # Set critic_updates, i.e. the times the critic gets trained w.r.t. one training step of the generator
        self.critic_updates = 5
        # Set weight of gradient penalty
        self.gp_weight = 10.0

    @tf.function
    def gradient_penalty(self,
                         batch_size: int,
                         real_imgs: tf.Tensor,
                         fake_imgs: tf.Tensor):
        # Get interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_imgs - real_imgs
        interpolated = real_imgs + alpha * diff

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
                   batch_size: int,
                   blurred_batch: tf.Tensor,
                   sharp_batch: tf.Tensor):
        d_losses = []
        # Train the critic multiple times according to critic_updates (by default, 5)
        for _ in range(self.critic_updates):
            with tf.GradientTape() as d_tape:
                # Generate fake inputs
                generated_batch = self.generator(blurred_batch, training=True)
                # Get logits for both fake and real images
                fake_logits = self.critic(generated_batch, training=True)
                real_logits = self.critic(sharp_batch, training=True)
                # Calculate critic's loss
                d_loss_fake = self.d_loss(tf.zeros_like(fake_logits), fake_logits)
                d_loss_real = self.d_loss(tf.ones_like(real_logits), real_logits)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                # Calculate gradient penalty
                gp = self.gradient_penalty(batch_size, blurred_batch, sharp_batch)
                # Add gradient penalty to the loss
                d_loss += gp * self.gp_weight
            # Get gradient w.r.t. critic's loss and update weights
            d_grad = d_tape.gradient(d_loss, self.critic.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_grad, self.critic.trainable_variables))

            d_losses.append(d_loss)

        # Train the generator
        with tf.GradientTape() as g_tape:
            # Generate fake inputs
            generated_batch = self.generator(blurred_batch, training=True)
            # Get logits for fake images
            fake_logits = self.critic(generated_batch, training=True)
            # Calculate generator's loss
            g_loss = self.g_loss(blurred_batch, sharp_batch)
        # Get gradient w.r.t. generator's loss and update weights
        g_grad = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grad, self.generator.trainable_variables))

        # Compute metrics
        ssim_metric = ssim(sharp_batch, generated_batch)
        psnr_metric = psnr(sharp_batch, generated_batch)

        return {"d_loss": np.mean(d_losses),
                "g_loss": g_loss,
                "ssim": ssim_metric,
                "psnr": psnr_metric}
