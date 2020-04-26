import tensorflow as tf
import numpy as np
import glob
from datasets import cifar_dataset
import math

from model.unet20 import UNet20
from model.res_skip_unet import ResSkipUNet
from datasets.reds_dataset import  load_image_dataset

def ssim(trueY, predY):
    return tf.image.ssim(trueY, predY, max_val=1.)

def psnr(trueY, predY):
    return tf.image.psnr(trueY, predY, max_val=1.)

# reds
"""
_, _, test = load_image_dataset("/home/uni/dataset/", 21000, 3000, [288, 512], 4, 10, 42)
test = test.batch(4)

weights = glob.glob('/home/uni/weights/reds/ep:020-val_loss:0.538.hdf5')

conv_net = UNet20(input_shape=(288, 512, 3))
conv_net.model.load_weights(weights[0])
"""

# cifar
(_, _), (_, _), (testX_64, testY_64) = cifar_dataset.load_image_dataset(normalize=True)

testX = tf.cast(testX_64, tf.float32)
testY = tf.cast(testY_64, tf.float32)

testX = tf.data.Dataset.from_tensor_slices(testX)
testY = tf.data.Dataset.from_tensor_slices(testY)

test = tf.data.Dataset.zip((testX, testY))
test = test.batch(512)
weights = glob.glob('/content/drive/My Drive/cnn-deblur/resskipunet/ep:049-val_loss:-30.774.hdf5')

conv_net = ResSkipUNet(input_shape=(32, 32, 3))
conv_net.model.load_weights(weights[0])

ssims = []
psnrs = []

for pair in test:
    x = pair[0]
    y = pair[1]

    predicted = tf.convert_to_tensor(conv_net.predict(x))

    delta_s = ssim(predicted, y) - ssim(x, y)
    delta_p = psnr(predicted, y) - psnr(x, y)

    ssims += delta_s.numpy().tolist()
    psnrs += delta_p.numpy().tolist()

avg_ssim = sum(ssims) / len(ssims)
# remove -inf elements
psnrs_clean = [x for x in psnrs if x != -math.inf]
avg_psnr = sum(psnrs_clean) / len(psnrs_clean)

print("Average improvement SSIM: {}".format(avg_ssim))
print("Average improvement PSNR: {}".format(avg_psnr))