from tensorflow.keras.datasets import cifar10
import cv2
import numpy as np
from datasets.dataset_utils import load_dataset_from_gcs
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Optional


def load_data(batch_size: int,
              epochs: int,
              val_size: int,
              seed: Optional[int] = 42):
    return load_dataset_from_gcs(project_id='cnn-deblur',
                                 bucket_name='cnn-d3blur-buck3t',
                                 prefix='cifar10',
                                 res=(32, 32),
                                 val_size=val_size,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 seed=seed)


def load_image_dataset(do_val_split: Optional[bool] = True, normalize: Optional[bool] = False):
    """Function that loads Cifar10 dataset and produces a training and test set in which the predictors are randomly
    Gaussian blurred images and the targets are the clear version of such images.
        :param do_val_split: boolean indicating whether the validation split must be performed or not (default is True)
        :param normalize: boolean indicating whether the pixel values should be normalized between 0 and 1 (optional)

        :return train: tuple containing predictor and target images of the train set
        :return test: tuple containing predictor and target images of the test set"""

    # Load training and test sets from Cifar10 dataset (labels are ignored)
    (train_set, _), (test_set, _) = cifar10.load_data()

    # Set random state to ensure reproducible results and blur the dataset
    rnd = np.random.RandomState(seed=42)
    (trainX, trainY), (testX, testY) = blur_dataset(train_set, test_set, normalize, rnd)

    # Reserve some samples for validation, if specified
    if do_val_split:
        trainX, valX, trainY, valY = train_test_split(trainX, trainY, random_state=rnd)
        return (trainX, trainY), (testX, testY), (valX, valY)
    else:
        return (trainX, trainY), (testX, testY)


def blur_dataset(train_set: np.ndarray,
                 test_set: np.ndarray,
                 normalize: Optional[bool] = False,
                 rnd: Optional[np.random.RandomState] = None):
    """Function which concurrently blurs a training and a test datasets by applying random Gaussian noise
        :param train_set: NumPy array representing the training set (clean images)
        :param test_set: NumPy array representing the test set (clean images)
        :param normalize: boolean flag which determines whether the pixel values will be normalized between 0 and 1
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
    if normalize:
        trainX = trainX.astype(np.float) / 255
        trainY = trainY.astype(np.float) / 255
        testX = testX.astype(np.float) / 255
        testY = testY.astype(np.float) / 255

    return (trainX, trainY), (testX, testY)
