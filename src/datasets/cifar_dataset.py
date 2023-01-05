import time
import typing
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

from .dataset_utils import load_dataset_from_gcs


def load_data(
        batch_size: int,
        epochs: int,
        val_size: int,
        seed: int = 42,
        repeat: bool = True,
        zero_mean: bool = False
):
    return load_dataset_from_gcs(
        project_id='cnn-deblur',
        bucket_name='cnn-d3blur-buck3t',
        prefix='cifar10',
        res=(32, 32),
        val_size=val_size,
        batch_size=batch_size,
        epochs=epochs,
        seed=seed,
        repeat=repeat,
        zero_mean=zero_mean
    )


def load_image_dataset(val_ratio: float = 0.125, normalization: int = 1):
    """Function that loads Cifar10 dataset and produces a training and test set in which the predictors are randomly
    Gaussian blurred images and the targets are the clear version of such images.
        :param val_ratio: boolean indicating the ratio of the validation split (if zero, the validation split
        is not performed)
        :param normalization: integer indicating the normalization type: 0 -> none, 1 -> [0, 1] (default), 2 -> [-1, 1]

        :return train: tuple containing predictor and target images of the train set
        :return test: tuple containing predictor and target images of the test set"""

    # Load training and test sets from Cifar10 dataset (labels are ignored)
    (train_set, _), (test_set, _) = keras.datasets.cifar10.load_data()

    # Set random state to ensure reproducible results and blur the dataset
    rnd = np.random.RandomState(seed=42)
    (trainX, trainY), (testX, testY) = blur_dataset(train_set, test_set, normalization, rnd)

    # Reserve some samples for validation, if specified
    if val_ratio != 0:
        trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=val_ratio, random_state=rnd)
        return (trainX, trainY), (testX, testY), (valX, valY)
    else:
        return (trainX, trainY), (testX, testY)


def blur_dataset(
        train_set: np.ndarray,
        test_set: np.ndarray,
        normalization: int = 1,
        rnd: typing.Optional[np.random.RandomState] = None
):
    """Function which concurrently blurs a training and a test datasets by applying random Gaussian noise
        :param train_set: NumPy array representing the training set (clean images)
        :param test_set: NumPy array representing the test set (clean images)
        :param normalization: integer indicating the normalization type: 0 -> none, 1 -> [0, 1] (default), 2 -> [-1, 1]
        :param rnd: random state to ensure reproducible results (optional)

        :returns the training set divided in predictor and target images
        :returns the test set divided in predictor and target images"""
    trainX = None
    trainY = None
    testX = None
    testY = None

    # Function to pass to the ThreadExecutor
    def _blur_dataset_thread(target: np.ndarray) -> np.ndarray:
        """Function which, given the target composed of the clear images, computes the predictor by applying random
        gaussian blur
            :param target: set of clear images

            :returns the set of blurred images"""
        subset_size = target.shape[0]

        # Function which blurs a given image with Gaussian blur
        # (standard deviation chosen randomly between 0 and 3)
        def _gauss_blur(img):
            if rnd is not None:
                std_dev = rnd.uniform(0, 3)
            else:
                std_dev = np.random.uniform(0, 3)
            return cv2.GaussianBlur(src=img, ksize=(0, 0), sigmaX=std_dev, borderType=cv2.BORDER_DEFAULT)

        # Create predictor
        predictor = np.zeros(shape=target.shape, dtype=target.dtype)

        # Save in predictor the blurred version of the target
        for i in range(subset_size):
            predictor[i] = _gauss_blur(target[i])

        return predictor

    # Concurrently produce train and test sets
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futureTrain = executor.submit(_blur_dataset_thread, train_set)
        futureTest = executor.submit(_blur_dataset_thread, test_set)
        futures = [futureTrain, futureTest]
        for future in as_completed(futures):
            if future == futureTrain:
                trainX = future.result()
                trainY = train_set
            else:
                testX = future.result()
                testY = test_set
    print('Time elapsed: {0:.2f} s'.format(time.time() - start_time))

    # Normalize if required
    if normalization == 1:
        trainX = trainX / 255.
        trainY = trainY / 255.
        testX = testX / 255.
        testY = testY / 255.
    elif normalization == 2:
        trainX = 2. * (trainX / 255.) - 1.
        trainY = 2 * (trainY / 255.) - 1.
        testX = 2 * (testX / 255.) - 1.
        testY = 2 * (testY / 255.) - 1.

    return (trainX, trainY), (testX, testY)
