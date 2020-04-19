import tensorflow as tf
import numpy as np
import glob
from datasets import cifar_dataset

from model.unet20 import UNet20
from model.u_net import UNet
from datasets.reds_dataset import  load_image_dataset

def ssim(trueY, predY):
    return tf.image.ssim(trueY, predY, max_val=1.)

# reds
"""
_, _, test = load_image_dataset("/home/uni/dataset/", 21000, 3000, [288, 512], 4, 10, 42)
test = test.batch(1)

weights = glob.glob('/home/uni/weights/reds/ep:020-val_loss:0.538.hdf5')

conv_net = UNet20(input_shape=(288, 512, 3))
conv_net.compile(loss='mae')
conv_net.model.load_weights(weights[0])
"""

# cifar
(_, _), (_, _), (testX_64, testY_64) = cifar_dataset.load_image_dataset(normalize=True)

testX = tf.cast(testX_64, tf.float32)
testY = tf.cast(testY_64, tf.float32)

testX = tf.data.Dataset.from_tensor_slices(testX)
testY = tf.data.Dataset.from_tensor_slices(testY)

test = tf.data.Dataset.zip((testX, testY))
test = test.batch(1)
print(test.take(1))
weights = glob.glob('/home/uni/weights/cifar/ep:050-val_loss:-31.109.hdf5')

conv_net = UNet(input_shape=(32, 32, 3))
conv_net.compile(loss='mae')
conv_net.model.load_weights(weights[0])

ssims = []

i = 0
for pair in test:
    i += 1
    print(i)

    x = pair[0]
    y = pair[1]

    predicted = tf.convert_to_tensor(conv_net.predict(x))

    delta = ssim(predicted, y) - ssim(x, y)
    ssims.append(delta)

avg = sum(ssims) / len(ssims)

print("Average improvement: {}".format(avg))