from tensorflow.keras.datasets import cifar10
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Optional


# ---------- CIFAR10 ----------
def preproc_cifar10(res: Optional[Tuple[int, int]] = None, normalize: Optional[bool] = False):
    """Function that loads Cifar10 dataset and produces a training and test set in which the predictors are randomly
    Gaussian blurred images and the targets are the clear version of such images.
        :param res: tuple representing the desired resolution (optional)
        :param normalize: boolean indicating whether the pixel values should be normalized between 0 and 1 (optional)

        :return train: tuple containing predictor and target images of the train set
        :return test: tuple containing predictor and target images of the test set"""

    # Load training and test sets from Cifar10 dataset (labels are ignored)
    (train_set, _), (test_set, _) = cifar10.load_data()

    # Set random state to ensure reproducible results and blur the dataset
    rnd = np.random.RandomState(seed=42)
    (trainX, trainY), (testX, testY) = blur_dataset(train_set, test_set, normalize, rnd)

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

    # Concurrently produce train and test sets
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futureTrain = executor.submit(blur_dataset_thread, train_set, rnd)
        futureTest = executor.submit(blur_dataset_thread, test_set, rnd)
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


def blur_dataset_thread(target: np.ndarray,
                        rnd: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Function which, given the target composed of the clear images, computes the predictor by applying random
    gaussian blur
        :param target: set of clear images
        :param rnd: random state to ensure reproducible results (optional)

        :returns the set of blurred images"""
    subset_size = target.shape[0]

    # Function which blurs a given image with Gaussian blur
    # (standard deviation chosen randomly between 0 and 3)
    def gauss_blur(img):
        if rnd is not None:
            std_dev = rnd.uniform(0, 3)
        else:
            std_dev = np.random.uniform(0, 3)
        return cv2.GaussianBlur(src=img, ksize=(0, 0), sigmaX=std_dev, borderType=cv2.BORDER_DEFAULT)

    # Create predictor
    predictor = np.zeros(shape=target.shape, dtype=target.dtype)

    # Save in predictor the blurred version of the target
    for i in range(subset_size):
        predictor[i] = gauss_blur(target[i])

    return predictor


def upscale_pad_dataset(trainX: np.ndarray,
                        trainY: np.ndarray,
                        testX: np.ndarray,
                        testY: np.ndarray,
                        res: Tuple[int, int]):
    """Function which upscales the dataset to half of the given resolution, and then adds padding """
    return (trainX, trainY), (testX, testY)


# ---------- REDS ----------
def resize_from_folder(input_folder, output_folder, new_dimensions):
    """Function that reads all the files in the input folder, resize them to match the specified (width, height)
    and finally store them in the output folder
        :param input_folder: string indicating the path of the input folder
        :param output_folder: string indicating the path of the output folder
        :param new_dimensions: tuple indicating the desired width and height in pixel of the resized images
    """

    # !Attention! all the files in the folder will be considered
    only_files = [f for f in listdir(input_folder) if isfile(join(input_folder, f))]

    for filename in only_files:
        full_path_input = join(input_folder, filename)
        img = cv2.imread(full_path_input, cv2.IMREAD_UNCHANGED)

        resized = cv2.resize(img, new_dimensions, interpolation = cv2.INTER_AREA)

        full_path_output = join(output_folder, filename)
        if not cv2.imwrite(full_path_output, resized):
            print("[ERROR] Impossible to save resized image {}".format(full_path_output))
