import tensorflow as tf


def ssim(trueY, predY):
    return tf.image.ssim(trueY, predY, max_val=1.)


def psnr(trueY, predY):
    return tf.image.psnr(trueY, predY, max_val=1.)
