import tensorflow as tf
from glob import glob
import os
from tqdm import tqdm

BASE_DIR = '/run/media/nihil/Backup/REDS/val'
BLUR_BASE = '/run/media/nihil/Backup/REDS/val/val_blur'
SHARP_BASE = '/run/media/nihil/Backup/REDS/val/val_sharp'


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(blur_path, sharp_path):
    blur_buf = open(blur_path, 'rb').read()
    sharp_buf = open(sharp_path, 'rb').read()

    return tf.train.Example(features=tf.train.Features(feature={
        'blur': _bytes_feature(blur_buf),
        'sharp': _bytes_feature(sharp_buf)
    }))


def main():
    blur_subdirs = glob(os.path.join(BLUR_BASE, '*', ''))
    sharp_subdirs = glob(os.path.join(SHARP_BASE, '*', ''))

    for blur_sub, sharp_sub in tqdm(zip(blur_subdirs, sharp_subdirs)):
        sub_name = os.path.basename(os.path.normpath(blur_sub))
        blur_batch = glob(os.path.join(blur_sub, '*.png'))
        sharp_batch = glob(os.path.join(sharp_sub, '*.png'))

        with tf.io.TFRecordWriter(os.path.join(BASE_DIR, 'REDS{0}.tfrecords'.format(sub_name))) as writer:
            for blur_img, sharp_img in tqdm(zip(blur_batch, sharp_batch)):
                example = _convert_to_example(blur_img, sharp_img)
                writer.write(example.SerializeToString())


if __name__ == '__main__':
    main()
