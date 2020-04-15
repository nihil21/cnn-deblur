import tensorflow as tf
from glob import glob
import os
from tqdm import tqdm


def reds_to_tfrecords(out_dir, blur_base, sharp_base):
    # Get lists of blur and sharp subdirectories
    blur_subdirs = glob(os.path.join(blur_base, '*', ''))
    sharp_subdirs = glob(os.path.join(sharp_base, '*', ''))

    # Define function to convert a string into a list of bytes
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    # Define a function to convert the pair of blurred and sharp images into a Tensorflow Example object
    def _convert_to_example(blur_path, sharp_path):
        blur_buf = open(blur_path, 'rb').read()
        sharp_buf = open(sharp_path, 'rb').read()

        return tf.train.Example(features=tf.train.Features(feature={
            'blur': _bytes_feature(blur_buf),
            'sharp': _bytes_feature(sharp_buf)
        }))

    # Iterate over blur and sharp subdirectories
    for blur_sub, sharp_sub in tqdm(zip(blur_subdirs, sharp_subdirs)):
        sub_name = os.path.basename(os.path.normpath(blur_sub))
        blur_batch = glob(os.path.join(blur_sub, '*.png'))
        sharp_batch = glob(os.path.join(sharp_sub, '*.png'))

        # For each subdirectory, create a .tfrecords file where Examples objects are stored
        with tf.io.TFRecordWriter(os.path.join(out_dir, 'REDS{0}.tfrecords'.format(sub_name))) as writer:
            for blur_img, sharp_img in tqdm(zip(blur_batch, sharp_batch)):
                example = _convert_to_example(blur_img, sharp_img)
                writer.write(example.SerializeToString())
