# -*- coding: utf-8 -*-
"""cnn-deblur

# cnn-deblur

This project aims at developing a Deep Neural Network able to deblur images, as part of the **Deep Learning** cours of the **Master in Artificial Intelligence** (*Alma Mater Studiorum*).  
This DNN should be able to correct both gaussian and motion blur, by training on Cifar10 and REDS datasets.

#### Authors:
 - Mattia Orlandi
 - Giacomo Pinardi

## Premises

## Prepare callback function to save weights.
"""

import argparse
import tensorflow as tf
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils.image_preprocessing import *
from model.conv_net import ConvNet
from model.u_net_reds import UNetREDS
from model.unet20 import UNet20
from model.toy_resnet import ToyResNet
from model.resnet_64_dense import ResNet64Dense
from model.resnet_64 import ResNet64
from model.resnet_128 import ResNet128

arch_choices = ['toy', '64dense', '64', '128', 'unetreds', 'unet20']

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--architecture", required=True, choices=arch_choices,
                help="architecture type [toy|64dense|64|128|unetreds|unet20]")
ap.add_argument("-ie", "--initial-epoch", required=True, help="initial epoch for the training process")
ap.add_argument("-fe", "--final-epoch", required=True, help="final epoch for the training process")
ap.add_argument("-bs", "--batch-size", required=True, help="batch-size dimension")
ap.add_argument("-l", "--loss", required=True, help="the loss function to use [mse|mae|...]")
args = vars(ap.parse_args())

filepath = '/home/uni/weights/reds/ep:{epoch:03d}-val_loss:{val_loss:.3f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

"""Enable GPU to speed up the processing; in case the notebook is not connected to a GPU runtime, it falls back to TPU.  
If that fails too, an exception is raised.
"""

device_name = tf.test.gpu_device_name()
print('Found GPU at: {}'.format(device_name))

# Change working directory
os.chdir('/home/uni/cnn-deblur/cnn-deblur')

# Create dictionary of possible architectures
arch_dict = dict({
    'toy': ToyResNet,
    '64dense': ResNet64Dense,
    '64': ResNet64,
    '128': ResNet128,
    'unetreds': UNetREDS,
    'unet20': UNet20
})
arch_type = args['architecture']

# Useful functions
def load_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image)
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image


def resize_image(image, new_height, new_width):
    return tf.image.resize(image, [new_height, new_width])


def random_flip(image_blur, image_sharp, seed):
    do_flip = tf.random.uniform([], seed=seed) > 0.5
    image_blur = tf.cond(do_flip, lambda: tf.image.flip_left_right(image_blur), lambda: image_blur)
    image_sharp = tf.cond(do_flip, lambda: tf.image.flip_left_right(image_sharp), lambda: image_sharp)

    return image_blur, image_sharp


# Create Dataset objects in order to lazily fetch the images
init_ep = int(args['initial_epoch'])
total_ep = int(args['final_epoch'])

EPOCHS = total_ep - init_ep
BATCH_SIZE = int(args['batch_size'])

seed = 42
new_dimension = [288, 512]
train_val_elements = 24000
validation_split = 0.125

# Training and validation sets
blur = tf.data.Dataset.list_files('/home/uni/dataset/train/train_blur/*/*', shuffle=True, seed=seed)
sharp = tf.data.Dataset.list_files('/home/uni/dataset/train/train_sharp/*/*', shuffle=True, seed=seed)

blur = blur.map(lambda filename: load_image(filename), num_parallel_calls=tf.data.experimental.AUTOTUNE)
sharp = sharp.map(lambda filename: load_image(filename), num_parallel_calls=tf.data.experimental.AUTOTUNE)

blur = blur.map(lambda image: resize_image(image, new_dimension[0], new_dimension[1]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
sharp = sharp.map(lambda image: resize_image(image, new_dimension[0], new_dimension[1]), num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset = tf.data.Dataset.zip((blur, sharp))
# reshuffle_each_iteration=False ensures that train and validation set are disjoint
dataset = dataset.shuffle(buffer_size=50, seed=seed, reshuffle_each_iteration=False)

# train and validation split
train = dataset.skip(int(train_val_elements*validation_split))
validation = dataset.take(int(train_val_elements*validation_split))

# data augmentation
train_augmented = train.map(lambda image_blur, image_sharp: random_flip(image_blur, image_sharp, seed), num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_augmented = train_augmented.shuffle(buffer_size=50, seed=seed, reshuffle_each_iteration=True)

# Cache datasets
train_augmented = train_augmented.cache()
validation = validation.cache()

# repeat: once for each epoch
train_augmented = train_augmented.batch(BATCH_SIZE).repeat(EPOCHS)
validation = validation.batch(BATCH_SIZE).repeat(EPOCHS)

train_augmented.prefetch(tf.data.experimental.AUTOTUNE)
validation.prefetch(tf.data.experimental.AUTOTUNE)

# Test set
test_elements = 3000

blur_test = tf.data.Dataset.list_files('/home/uni/dataset/val/val_blur/*/*', shuffle=False)
sharp_test = tf.data.Dataset.list_files('/home/uni/dataset/val/val_sharp/*/*', shuffle=False)

blur_test = blur_test.map(lambda filename: load_image(filename))
sharp_test = sharp_test.map(lambda filename: load_image(filename))

blur_test = blur_test.map(lambda image: resize_image(image, new_dimension[0], new_dimension[1]))
sharp_test = sharp_test.map(lambda image: resize_image(image, new_dimension[0], new_dimension[1]))

test = tf.data.Dataset.zip((blur_test, sharp_test))

# Create ConvNet and plot model
conv_net = arch_dict[arch_type](input_shape=(288, 512, 3))
loss_fun = args['loss']
conv_net.compile(loss=loss_fun)

print(conv_net.summary())
path_to_graphs = os.path.join('..', 'res')
conv_net.plot_model(os.path.join(path_to_graphs, 'model.png'))

"""Load weights from previous run."""
if init_ep != 0:
    weights = glob.glob('/home/uni/weights/reds/ep:{0:03d}-val_loss:*.hdf5'.format(init_ep))
    conv_net.model.load_weights(weights[0])
    print('Initial epoch: {0:d}'.format(init_ep))

"""Train model following *train-validation-test* paradigm."""
steps_train = int((train_val_elements*(1-validation_split)) // BATCH_SIZE)
steps_val = int((train_val_elements*validation_split) // BATCH_SIZE)

hist = conv_net.fit(train_augmented,
                    epochs=total_ep,
                    steps_per_epoch=steps_train,
                    validation_data=validation,
                    validation_steps=steps_val,
                    initial_epoch=init_ep,
                    callbacks=callbacks_list)

"""Evaluate the model on the test set."""

results = conv_net.model.evaluate(test)
print('Test loss:', results[0])
print('Test ssim_metric:', results[1])
print('Test mse:', results[2])
print('Test mae:', results[3])
print('Test mape:', results[4])
print('Test cosine_proximity:', results[5])

"""Plot graph representing the loss and accuracy trends over epochs."""

n = np.arange(0, total_ep - init_ep)
plt.style.use('ggplot')
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 16))
fig.suptitle('Loss and metrics trends over epochs')
# Loss
axes[0, 0].plot(n, hist.history['loss'], label='train_loss')
axes[0, 0].plot(n, hist.history['val_loss'], label='val_loss')
axes[0, 0].set_title('Loss')
axes[0, 0].set(xlabel='Epochs #', ylabel='Loss')
axes[0, 0].legend()
# SSIM
axes[0, 1].plot(n, hist.history['ssim_metric'], label='train_ssim_metric')
axes[0, 1].plot(n, hist.history['val_ssim_metric'], label='val_ssim_metric')
axes[0, 1].set_title('SSIM')
axes[0, 1].set(xlabel='Epochs #', ylabel='SSIM')
axes[0, 1].legend()
# MSE
axes[1, 0].plot(n, hist.history['mse'], label='train_mse')
axes[1, 0].plot(n, hist.history['val_mse'], label='val_mse')
axes[1, 0].set_title('MSE')
axes[1, 0].set(xlabel='Epochs #', ylabel='MSE')
axes[1, 0].legend()
# MAE
axes[1, 1].plot(n, hist.history['mae'], label='train_mae')
axes[1, 1].plot(n, hist.history['val_mae'], label='val_mae')
axes[1, 1].set_title('MAE')
axes[1, 1].set(xlabel='Epochs #', ylabel='MAE')
axes[1, 1].legend()
# MAPE
axes[2, 0].plot(n, hist.history['mape'], label='train_mape')
axes[2, 0].plot(n, hist.history['val_mape'], label='val_mape')
axes[2, 0].set_title('MAPE')
axes[2, 0].set(xlabel='Epochs #', ylabel='MAPE')
axes[2, 0].legend()
# Cosine Proximity
axes[2, 1].plot(n, hist.history['cosine_proximity'], label='train_cosine_proximity')
axes[2, 1].plot(n, hist.history['val_cosine_proximity'], label='val_cosine_proximity')
axes[2, 1].set_title('Cosine Proximity')
axes[2, 1].set(xlabel='Epochs #', ylabel='Cosine Proximity')
axes[2, 1].legend()

fig.savefig(os.path.join(path_to_graphs, 'metrics.png'))

"""Generate predictions on new data."""

blurred = test.take(3)
original = test.take(3)

predicted = conv_net.predict(blurred)

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
for i in range(3):
    axes[i, 0].set_title('Original image')
    axes[i, 0].imshow(original[i])
    axes[i, 1].set_title('Blurred image')
    axes[i, 1].imshow(blurred[i])
    axes[i, 2].set_title('Predicted image')
    axes[i, 2].imshow(predicted[i])

fig.savefig(os.path.join(path_to_graphs, 'predictions.png'))

