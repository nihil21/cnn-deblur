import tensorflow as tf
import os
from glob import glob


def load_image_dataset(dataset_root,
                       train_size,
                       val_size,
                       new_res,
                       batch_size,
                       epochs,
                       seed):
    # Training and validation sets
    blur = tf.data.Dataset.list_files(os.path.join(dataset_root,
                                                   'train', 'train_blur', '*', '*'), shuffle=True, seed=seed)
    sharp = tf.data.Dataset.list_files(os.path.join(dataset_root,
                                                    'train', 'train_sharp', '*', '*'), shuffle=True, seed=seed)

    # Define function to load images and map it
    def _load_image(filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_png(image)
        image = tf.image.convert_image_dtype(image, tf.float32)

        return image

    blur = blur.map(lambda filename: _load_image(filename),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    sharp = sharp.map(lambda filename: _load_image(filename),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Define function to resize images and map it
    def _resize_image(image, new_height, new_width):
        return tf.image.resize(image, [new_height, new_width])

    blur = blur.map(lambda image: _resize_image(image, new_res[0], new_res[1]),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    sharp = sharp.map(lambda image: _resize_image(image, new_res[0], new_res[1]),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = tf.data.Dataset.zip((blur, sharp))

    # reshuffle_each_iteration=False ensures that train and validation set are disjoint
    dataset = dataset.shuffle(buffer_size=50, seed=seed, reshuffle_each_iteration=False)

    # train and validation split
    train = dataset.skip(train_size)
    validation = dataset.take(val_size)

    # data augmentation
    def _random_flip(image_blur, image_sharp):
        do_flip = tf.random.uniform([], seed=seed) > 0.5
        image_blur = tf.cond(do_flip, lambda: tf.image.flip_left_right(image_blur), lambda: image_blur)
        image_sharp = tf.cond(do_flip, lambda: tf.image.flip_left_right(image_sharp), lambda: image_sharp)

        return image_blur, image_sharp

    train_augmented = train.map(_random_flip,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_augmented = train_augmented.shuffle(buffer_size=50, seed=seed, reshuffle_each_iteration=True)

    # Cache datasets
    train_augmented = train_augmented.cache()
    validation = validation.cache()

    # repeat: once for each epoch
    train_augmented = train_augmented.batch(batch_size).repeat(epochs)
    validation = validation.batch(batch_size).repeat(epochs)

    # Prefetch
    train_augmented.prefetch(tf.data.experimental.AUTOTUNE)
    validation.prefetch(tf.data.experimental.AUTOTUNE)

    # Test set
    blur_test = tf.data.Dataset.list_files(os.path.join(dataset_root,
                                                        'val', 'val_blur', '*', '*'), shuffle=False)
    sharp_test = tf.data.Dataset.list_files(os.path.join(dataset_root,
                                                         'val', 'val_sharp', '*', '*'), shuffle=False)

    blur_test = blur_test.map(lambda filename: _load_image(filename))
    sharp_test = sharp_test.map(lambda filename: _load_image(filename))
    blur_test = blur_test.map(lambda image: _resize_image(image, new_res[0], new_res[1]))
    sharp_test = sharp_test.map(lambda image: _resize_image(image, new_res[0], new_res[1]))
    test = tf.data.Dataset.zip((blur_test, sharp_test))

    return train_augmented, validation, test


def load_tfrecord_dataset(dataset_root,
                          train_size,
                          val_size,
                          new_res,
                          batch_size,
                          epochs,
                          seed):
    BUF = 50

    # Load .tfrecords files
    tf_trainval = glob(os.path.join(dataset_root, 'train', '*.tfrecords'))
    trainval_data = tf.data.TFRecordDataset(filenames=tf_trainval,
                                            num_parallel_reads=tf.data.experimental.AUTOTUNE)
    tf_test = glob(os.path.join(dataset_root, 'test', '*.tfrecords'))
    test_data = tf.data.TFRecordDataset(filenames=tf_test,
                                        num_parallel_reads=tf.data.experimental.AUTOTUNE)

    image_features_dict = {
        'blur': tf.io.FixedLenFeature([], tf.string),
        'sharp': tf.io.FixedLenFeature([], tf.string)
    }

    # Define function to parse tfrecords as image pairs (blur + sharp)
    def _parse_image_fn(proto):
        images = tf.io.parse_single_example(proto, image_features_dict)

        blur_img = tf.image.decode_png(images['blur'], channels=3)
        blur_img = tf.image.resize(blur_img, new_res)
        blur_img /= 255.0
        sharp_img = tf.image.decode_png(images['sharp'], channels=3)
        sharp_img = tf.image.resize(sharp_img, new_res)
        sharp_img /= 255.0

        return blur_img, sharp_img

    # Map parsing function
    trainval_data = trainval_data.map(_parse_image_fn,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_data = test_data.map(_parse_image_fn,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle once and perform train-validation split
    trainval_data = trainval_data.shuffle(buffer_size=50, seed=seed, reshuffle_each_iteration=False)
    train_data = trainval_data.skip(val_size)
    val_data = trainval_data.take(train_size)

    train_data = train_data.batch(batch_size)
    val_data = val_data.batch(batch_size)
    test_data = test_data.batch(batch_size)

    def _random_flip(image_blur, image_sharp):
        do_flip = tf.random.uniform([], seed=seed) > 0.5
        image_blur = tf.cond(do_flip, lambda: tf.image.flip_left_right(image_blur), lambda: image_blur)
        image_sharp = tf.cond(do_flip, lambda: tf.image.flip_left_right(image_sharp), lambda: image_sharp)

        return image_blur, image_sharp

    # Perform augmentation on train set only and reshuffle
    train_data = train_data.map(_random_flip,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(buffer_size=BUF,
                                                                                          seed=seed,
                                                                                          reshuffle_each_iteration=True)

    # Repeat once for each epoch
    train_data = train_data.repeat(epochs)
    val_data = val_data.repeat(epochs)
    test_data = test_data.repeat()

    # Prefetch
    train_data.prefetch(tf.data.experimental.AUTOTUNE)
    val_data.prefetch(tf.data.experimental.AUTOTUNE)
    test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)

    return train_data, val_data, test_data
