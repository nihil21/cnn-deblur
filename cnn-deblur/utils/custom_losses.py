import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.losses import mse
import functools
import operator
from typing import Optional, List


def wasserstein_loss(trueY, predY):
    return K.mean(trueY * predY)


def perceptual_loss(trueY, predY, loss_model: Optional[Model] = None):
    if loss_model is None:
        vgg = VGG16(include_top=False, weights='imagenet', input_shape=(None, None, 3))
        loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
        loss_model.trainable = False
    return K.mean(K.square(loss_model(trueY) - loss_model(predY)))


def ms_mse(sharp_pyramid: List[tf.Tensor],
           predicted_pyramid: List[tf.Tensor],
           num_scales: Optional[int] = 3):
    # Check input
    assert len(sharp_pyramid) == num_scales, 'The list \'trueY\' should contain {:d} elements'.format(num_scales)
    assert len(predicted_pyramid) == num_scales, 'The list \'predY\' should contain {:d} elements'.format(num_scales)

    loss = 0.
    for scale_trueY, scale_predY in zip(sharp_pyramid, predicted_pyramid):
        scale_shape = tf.shape(scale_trueY)[1:]
        norm_factor = functools.reduce(operator.mul, scale_shape, 1)
        scale_loss = tf.reduce_sum(mse(scale_trueY, scale_predY)) / norm_factor
        loss += scale_loss
    return 1./(2. * num_scales) * loss
