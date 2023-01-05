# -*- coding: utf-8 -*-
"""
cnn-deblur

This project aims at developing a Deep Neural Network able to deblur images, as part of the
Deep Learning course of the Master in Artificial Intelligence (Alma Mater Studiorum).
This DNN should be able to correct both gaussian and motion blur, by training on Cifar10 and REDS datasets.

Authors:
 - Mattia Orlandi
 - Giacomo Pinardi
"""

import argparse
import tensorflow as tf
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from tensorflow import keras
from models.resnet import ResNet16, ResNet16Dense, ResNet20
from models.unet import UNet16, UNet20
from models.res_unet import ResUNet16
from models.res_skip_unet import ResSkipUNet
from datasets.reds_dataset import load_image_dataset, load_tfrecord_dataset
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ARCH_CHOICES = ['resnet16', 'resnet16dense', 'resnet20', 'unet16', 'unet20', 'resunet16', 'resskipunet']


def main():
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--architecture", required=True, choices=ARCH_CHOICES,
                    help="architecture type [resnet16|resnet16dense|resnet20|unet16|unet20|resunet16|resskipunet]")
    ap.add_argument("-ie", "--initial-epoch", required=True, help="initial epoch for the training process")
    ap.add_argument("-fe", "--final-epoch", required=True, help="final epoch for the training process")
    ap.add_argument("-bs", "--batch-size", required=True, help="batch-size dimension")
    ap.add_argument("-l", "--loss", required=True, help="the loss function to use [mse|mae|...]")
    ap.add_argument("-tf", "--tfrecords", action="store_true", help="usage of tfrecords")
    args = vars(ap.parse_args())

    if args['tfrecords']:
        use_tfrecords = True
        dataset_root = '/mnt/REDS/'
    else:
        use_tfrecords = False
        dataset_root = '/home/uni/dataset/'

    filepath = '/home/uni/weights/reds/ep:{epoch:03d}-val_loss:{val_loss:.3f}.hdf5'
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    print('=' * 50)
    device_name = tf.test.gpu_device_name()
    print('Found GPU at: {}'.format(device_name))

    # Change working directory
    os.chdir('/home/uni/cnn-deblur/cnn-deblur')

    # Create dictionary of possible architectures
    arch_dict = dict({
        'resnet16': ResNet16,
        'resnet16dense': ResNet16Dense,
        'resnet20': ResNet20,
        'unet16': UNet16,
        'unet20': UNet20,
        'resunet': ResUNet16,
        'resskipunet': ResSkipUNet
    })
    arch_type = args['architecture']

    # Create Dataset objects in order to lazily fetch the images
    print('=' * 50)
    print('Preprocessing REDS dataset...')
    init_ep = int(args['initial_epoch'])
    total_ep = int(args['final_epoch'])

    EPOCHS = total_ep - init_ep
    BATCH_SIZE = int(args['batch_size'])

    NEW_RES = [288, 512]
    TRAINVAL_SIZE = 24000
    TEST_SIZE = 3000
    VAL_SPLIT = 0.125
    RND = 42

    val_size = int(VAL_SPLIT * TRAINVAL_SIZE)
    train_size = TRAINVAL_SIZE - val_size

    if use_tfrecords:
        train_data, test_data, val_data = load_tfrecord_dataset(dataset_root,
                                                                val_size,
                                                                NEW_RES,
                                                                BATCH_SIZE,
                                                                EPOCHS,
                                                                RND)
    else:
        train_data, test_data, val_data = load_image_dataset(dataset_root,
                                                             val_size,
                                                             NEW_RES,
                                                             BATCH_SIZE,
                                                             EPOCHS,
                                                             RND)

    print('Training set size: {0:d}'.format(train_size))
    print('Validation set size: {0:d}'.format(val_size))
    print('Test set size: {0:d}'.format(TEST_SIZE))

    # Create ConvNet and plot model
    conv_net = arch_dict[arch_type](input_shape=(288, 512, 3))
    loss_fun = args['loss']
    conv_net.compile(loss=loss_fun)

    print('=' * 50)
    print('Model architecture:')
    print(conv_net.summary())
    path_to_graphs = os.path.join('..', 'res')
    conv_net.plot_model(os.path.join(path_to_graphs, 'model.png'))

    # Load weights from previous run.
    if init_ep != 0:
        weights = glob.glob('/home/uni/weights/reds/ep:{0:03d}-val_loss:*.hdf5'.format(init_ep))
        conv_net.model.load_weights(weights[0])
        print('Initial epoch: {0:d}'.format(init_ep))

    # Train model following train-validation-test paradigm.
    print('=' * 50)
    print('Training model...')
    steps_train = train_size // BATCH_SIZE
    steps_val = val_size // BATCH_SIZE

    hist = conv_net.fit(train_data,
                        epochs=total_ep,
                        steps_per_epoch=steps_train,
                        validation_data=val_data,
                        validation_steps=steps_val,
                        initial_epoch=init_ep,
                        callbacks=callbacks_list)

    print('=' * 50)
    print('Evaluating model...')
    # Evaluate the model on the test set.
    steps_test = TEST_SIZE // BATCH_SIZE
    results = conv_net.evaluate(test_data, steps=steps_test)
    print('Test loss:', results[0])
    print('Test ssim_metric:', results[1])
    print('Test mse:', results[2])
    print('Test mae:', results[3])
    print('Test accuracy:', results[4])

    # Plot graph representing the loss and accuracy trends over epochs.
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
    # Accuracy
    axes[2, 0].plot(n, hist.history['accuracy'], label='train_accuracy')
    axes[2, 0].plot(n, hist.history['val_accuracy'], label='val_accuracy')
    axes[2, 0].set_title('Accuracy')
    axes[2, 0].set(xlabel='Epochs #', ylabel='Accuracy')
    axes[2, 0].legend()

    fig.savefig(os.path.join(path_to_graphs, 'metrics.png'))

    # Generate predictions on new data.
    blurred = None
    original = None
    predicted = None
    for batch in test_data.take(1):
        blurred = batch[0]
        original = batch[1]
        predicted = conv_net.predict(batch[0])

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    for i in range(4):
        axes[i, 0].set_title('Original image')
        axes[i, 0].imshow(original[i])
        axes[i, 1].set_title('Blurred image')
        axes[i, 1].imshow(blurred[i])
        axes[i, 2].set_title('Predicted image')
        axes[i, 2].imshow(predicted[i])

    fig.savefig(os.path.join(path_to_graphs, 'predictions.png'))


if __name__ == '__main__':
    main()
