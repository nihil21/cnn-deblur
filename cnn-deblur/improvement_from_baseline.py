import tensorflow as tf
from datasets import cifar_dataset
import math

from datasets.reds_dataset import load_image_dataset


def ssim(trueY, predY):
    return tf.image.ssim(trueY, predY, max_val=1.)


def psnr(trueY, predY):
    return tf.image.psnr(trueY, predY, max_val=1.)


def setup(dataset):
    if dataset == "cifar":
        (_, _), (_, _), (testX_64, testY_64) = cifar_dataset.load_image_dataset(normalize=True)

        testX = tf.cast(testX_64, tf.float32)
        testY = tf.cast(testY_64, tf.float32)

        testX = tf.data.Dataset.from_tensor_slices(testX)
        testY = tf.data.Dataset.from_tensor_slices(testY)

        test = tf.data.Dataset.zip((testX, testY))
        test = test.batch(512)

        return test

    elif dataset == "reds":
        _, _, test = load_image_dataset(dataset_root="/home/uni/dataset/",
                                        val_size=3000,
                                        new_res=[288, 512],
                                        batch_size=4,
                                        epochs=10,
                                        seed=42)
        test = test.batch(4)

        return test


def compute_improvement_from_baseline(dataset_name, model):
    # Model must be already initialized
    # conv_net = UNet20(input_shape=(288, 512, 3))
    # weights = glob.glob('/home/uni/weights/reds/ep:020-val_loss:0.538.hdf5')
    # conv_net.model.load_weights(weights[0])

    test_set = setup(dataset_name)

    ssims = []
    psnrs = []

    for pair in test_set:
        x = pair[0]
        y = pair[1]

        predicted = tf.convert_to_tensor(model.predict(x))

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
