import tensorflow as tf


def ssim_metric(trueY, predY):
    return tf.image.ssim(trueY, predY, max_val=1.)


def psnr_metric(trueY, predY):
    return tf.image.psnr(trueY, predY, max_val=1.)


def psnr_loss(trueY, predY):
    return -psnr_metric(trueY, predY)
