import tensorflow as tf
from glob import glob
import os
import matplotlib.pyplot as plt
import time

BASE_DIR = '/mnt/REDS/train'
EPOCHS = 10
BATCH_SIZE = 4
NEW_DIM = (288, 512)
REDS_SIZE = 24000
VAL_SPLIT = 0.125
BUF_SIZE = 1000
RND = 42


def show_batch(batch):
    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 2
    for i in range(1, columns * rows + 1):
        img = batch[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


def main():
    tfrecords = glob(os.path.join(BASE_DIR, '*.tfrecords'))
    reds_dataset = tf.data.TFRecordDataset(filenames=tfrecords, num_parallel_reads=5)

    image_features_dict = {
        'blur': tf.io.FixedLenFeature([], tf.string),
        'sharp': tf.io.FixedLenFeature([], tf.string)
    }

    def _parse_image_fn(example):
        images = tf.io.parse_single_example(example, image_features_dict)

        blur_img = tf.image.decode_png(images['blur'], channels=3)
        blur_img = tf.image.resize(blur_img, NEW_DIM)
        blur_img /= 255.0
        sharp_img = tf.image.decode_png(images['sharp'], channels=3)
        sharp_img = tf.image.resize(sharp_img, NEW_DIM)
        sharp_img /= 255.0

        return blur_img, sharp_img

    reds_dataset = reds_dataset.map(_parse_image_fn)

    # Shuffle once
    reds_dataset = reds_dataset.shuffle(buffer_size=BUF_SIZE, seed=RND, reshuffle_each_iteration=False)

    # Training and validation split
    train = reds_dataset.skip(int(VAL_SPLIT * REDS_SIZE))
    val = reds_dataset.take(int(VAL_SPLIT * REDS_SIZE))

    def _random_flip(image_blur, image_sharp):
        do_flip = tf.random.uniform([], seed=RND) > 0.5
        image_blur = tf.cond(do_flip, lambda: tf.image.flip_left_right(image_blur), lambda: image_blur)
        image_sharp = tf.cond(do_flip, lambda: tf.image.flip_left_right(image_sharp), lambda: image_sharp)

        return image_blur, image_sharp

    # Perform augmentation and reshuffle
    train = train.map(_random_flip).shuffle(buffer_size=BUF_SIZE, seed=RND, reshuffle_each_iteration=True)

    # Repeat once for each epoch
    train = train.batch(BATCH_SIZE).repeat(EPOCHS)
    val = val.batch(BATCH_SIZE).repeat(EPOCHS)

    # Prefetch
    train.prefetch(10)
    val.prefetch(10)

    for batch_train, batch_val in zip(train.take(1), val.take(1)):
        print(batch_train[0].shape, batch_val[0].shape)

        show_batch(batch_train[0])
        show_batch(batch_train[1])
        show_batch(batch_val[0])
        show_batch(batch_val[1])


if __name__ == '__main__':
    start_time = time.time()
    main()
    print('Time elapsed: {0:.2f} s'.format(time.time() - start_time))
