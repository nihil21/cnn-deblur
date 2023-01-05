import typing

import tensorflow as tf
from tensorflow import keras


def wasserstein_loss(trueY, predY):
    return keras.backend.mean(trueY * predY)


def perceptual_loss(trueY, predY, loss_model: typing.Optional[keras.models.Model] = None):
    if loss_model is None:
        vgg = keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(None, None, 3))
        loss_model = keras.models.Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
        loss_model.trainable = False
    return keras.backend.mean(keras.backend.square(loss_model(trueY) - loss_model(predY)))


def ms_mse(
        sharp_pyramid: typing.List[tf.Tensor],
        predicted_pyramid: typing.List[tf.Tensor],
        num_scales: int = 3
):
    # Check input
    assert len(sharp_pyramid) == num_scales, 'The list \'trueY\' should contain {:d} elements'.format(num_scales)
    assert len(predicted_pyramid) == num_scales, 'The list \'predY\' should contain {:d} elements'.format(num_scales)

    loss = 0.
    for scale_trueY, scale_predY in zip(sharp_pyramid, predicted_pyramid):
        scale_shape = tf.shape(scale_trueY)[1:]
        norm_factor = tf.cast(tf.reduce_prod(scale_shape), dtype='float32')
        scale_loss = tf.reduce_sum(keras.losses.mse(scale_trueY, scale_predY)) / norm_factor
        loss += scale_loss
    return 1./(2. * num_scales) * loss


def ms_logcosh(
        sharp_pyramid: typing.List[tf.Tensor],
        predicted_pyramid: typing.List[tf.Tensor],
        num_scales: int = 3
):
    # Check input
    assert len(sharp_pyramid) == num_scales, 'The list \'trueY\' should contain {:d} elements'.format(num_scales)
    assert len(predicted_pyramid) == num_scales, 'The list \'predY\' should contain {:d} elements'.format(num_scales)

    loss = 0.
    for scale_trueY, scale_predY in zip(sharp_pyramid, predicted_pyramid):
        scale_shape = tf.shape(scale_trueY)[1:]
        norm_factor = tf.cast(tf.reduce_prod(scale_shape), dtype='float32')
        scale_loss = tf.reduce_sum(keras.losses.logcosh(scale_trueY, scale_predY)) / norm_factor
        loss += scale_loss
    return 1./(2. * num_scales) * loss


def ms_perceptual(
        sharp_pyramid: typing.List[tf.Tensor],
        predicted_pyramid: typing.List[tf.Tensor],
        num_scales: int = 3,
        loss_model: typing.Optional[keras.models.Model] = None
):
    # Check input
    assert len(sharp_pyramid) == num_scales, 'The list \'trueY\' should contain {:d} elements'.format(num_scales)
    assert len(predicted_pyramid) == num_scales, 'The list \'predY\' should contain {:d} elements'.format(num_scales)

    loss = 0.
    for scale_trueY, scale_predY in zip(sharp_pyramid, predicted_pyramid):
        scale_shape = tf.shape(scale_trueY)[1:]
        norm_factor = tf.cast(tf.reduce_prod(scale_shape), dtype='float32')
        scale_loss = tf.reduce_sum(perceptual_loss(scale_trueY, scale_predY, loss_model)) / norm_factor
        loss += scale_loss
    return 1./(2. * num_scales) * loss
