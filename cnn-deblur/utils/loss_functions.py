import tensorflow as tf
from tensorflow.keras.losses import mean_absolute_error
import tensorflow.keras.backend as K


def ssim_loss(trueY, predY):
    return -tf.image.ssim(trueY, predY, max_val=2.)


def mix_loss(trueY, predY):
    alpha = 0.84
    return alpha * ssim_loss(trueY, predY) + (1 - alpha) * tf.math.reduce_mean(mean_absolute_error(trueY, predY))


def content_loss(trueY, predY):
    return 0.5 * K.sum(K.square(trueY - predY))


def psnr_loss(trueY, predY):
    return -tf.image.psnr(trueY, predY, max_val=1.)
