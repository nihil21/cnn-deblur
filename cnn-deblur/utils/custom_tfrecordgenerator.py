import tensorflow as tf
from glob import glob
import os
import matplotlib.pyplot as plt

BASE_DIR = '/run/media/nihil/Backup/REDS/val'
EPOCHS = 10
BATCH_SIZE = 4
OLD_DIM = [720, 1280]
NEW_DIM = (288, 512)


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
    print('Number of records to load: {0:d}'.format(len(tfrecords)))

    reds_dataset = tf.data.TFRecordDataset(filenames=tfrecords)
    print('TFRecords loaded from disk')

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

    print('Parsing done')

    reds_dataset = reds_dataset.shuffle(buffer_size=1000, seed=42)

    # repeat: once for each epoch
    batched_dataset = reds_dataset.batch(BATCH_SIZE).repeat(EPOCHS)

    for batch in batched_dataset.take(1):
        print(batch[0].shape)

        show_batch(batch[0])
        show_batch(batch[1])


if __name__ == '__main__':
    main()
