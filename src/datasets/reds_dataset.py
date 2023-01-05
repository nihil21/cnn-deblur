import os
from glob import glob

import tensorflow as tf
from .dataset_utils import load_dataset_from_gcs


def load_data(
        batch_size: int,
        epochs: int,
        val_size: int,
        seed: int = 42,
        repeat: bool = True,
        zero_mean: bool = False,
        low_res: bool = False
):
    if low_res:
        res = (144, 256)
    else:
        res = (288, 512)
    return load_dataset_from_gcs(
        project_id='cnn-deblur',
        bucket_name='cnn-d3blur-buck3t',
        prefix='REDS',
        res=res,
        val_size=val_size,
        batch_size=batch_size,
        epochs=epochs,
        seed=seed,
        use_patches=True,
        repeat=repeat,
        zero_mean=zero_mean
    )


def load_image_dataset(
        dataset_root,
        val_size,
        new_res,
        batch_size,
        epochs,
        seed
):
    # Training and validation sets
    blur = tf.data.Dataset.list_files(
        os.path.join(dataset_root, 'train', 'train_blur', '*', '*'),
        shuffle=True,
        seed=seed
    )
    sharp = tf.data.Dataset.list_files(
        os.path.join(dataset_root, 'train', 'train_sharp', '*', '*'),
        shuffle=True,
        seed=seed
    )

    # Define function to load images and map it
    def _load_image(filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_png(image)
        image = tf.image.convert_image_dtype(image, tf.float32)

        return image

    blur = blur.map(
        lambda filename: _load_image(filename),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    sharp = sharp.map(
        lambda filename: _load_image(filename),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # Define function to resize images and map it
    def _resize_image(image, new_height, new_width):
        return tf.image.resize(image, [new_height, new_width])

    blur = blur.map(
        lambda image: _resize_image(image, new_res[0], new_res[1]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    sharp = sharp.map(
        lambda image: _resize_image(image, new_res[0], new_res[1]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    def _extract_patches(img):
        img = tf.reshape(img, (1, 720, 1280, 3))
        # from the single image extract the 12 patches
        # with input shape 720x1280 each patch has shape 240x320
        patches = tf.image.extract_patches(
            images=img,
            sizes=[1, 240, 320, 1],
            strides=[1, 240, 320, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        patches = tf.reshape(patches, (12, 240, 320, 3))

        return patches

    # extract patches
    blur = blur.map(_extract_patches, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # now each element of the dataset has shape (12, 240, 320, 3)
    # un-batch in order to have each element of shape (1, 240, 320, 3)
    blur = blur.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))

    sharp = sharp.map(_extract_patches, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    sharp = sharp.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))

    dataset = tf.data.Dataset.zip((blur, sharp))

    # reshuffle_each_iteration=False ensures that train and validation set are disjoint
    dataset = dataset.shuffle(buffer_size=50, seed=seed, reshuffle_each_iteration=False)

    # train and validation split
    train = dataset.skip(val_size)
    validation = dataset.take(val_size)

    # data augmentation
    def _random_flip(image_blur, image_sharp):
        do_flip = tf.random.uniform([], seed=seed) > 0.5
        image_blur = tf.cond(do_flip, lambda: tf.image.flip_left_right(image_blur), lambda: image_blur)
        image_sharp = tf.cond(do_flip, lambda: tf.image.flip_left_right(image_sharp), lambda: image_sharp)

        return image_blur, image_sharp

    train_augmented = train.map(
        _random_flip,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

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
    blur_test = tf.data.Dataset.list_files(
        os.path.join(dataset_root, 'val', 'val_blur', '*', '*'),
        shuffle=False
    )
    sharp_test = tf.data.Dataset.list_files(
        os.path.join(dataset_root, 'val', 'val_sharp', '*', '*'),
        shuffle=False
    )

    blur_test = blur_test.map(lambda filename: _load_image(filename))
    sharp_test = sharp_test.map(lambda filename: _load_image(filename))
    blur_test = blur_test.map(lambda image: _resize_image(image, new_res[0], new_res[1]))
    sharp_test = sharp_test.map(lambda image: _resize_image(image, new_res[0], new_res[1]))
    blur_test = blur_test.map(_extract_patches)
    blur_test = blur_test.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    sharp_test = sharp_test.map(_extract_patches)
    sharp_test = sharp_test.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    test = tf.data.Dataset.zip((blur_test, sharp_test))
    test = test.batch(batch_size)

    return train_augmented, test, validation


def load_tfrecord_dataset(
        dataset_root,
        val_size,
        new_res,
        batch_size,
        epochs,
        seed
):
    BUF = 50

    # Load .tfrecords files
    tf_trainval = glob(os.path.join(dataset_root, 'train', '*.tfrecords'))
    trainval_data = tf.data.TFRecordDataset(
        filenames=tf_trainval,
        num_parallel_reads=tf.data.experimental.AUTOTUNE
    )
    tf_test = glob(os.path.join(dataset_root, 'test', '*.tfrecords'))
    test_data = tf.data.TFRecordDataset(
        filenames=tf_test,
        num_parallel_reads=tf.data.experimental.AUTOTUNE
    )

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
    trainval_data = trainval_data.map(_parse_image_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_data = test_data.map(_parse_image_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Unzip
    trainval_dataX = trainval_data.map(lambda x, y: x)
    trainval_dataY = trainval_data.map(lambda x, y: y)
    test_dataX = test_data.map(lambda x, y: x)
    test_dataY = test_data.map(lambda x, y: y)

    def _extract_patches(image):
        image = tf.reshape(image, (1, 720, 1280, 3))
        # from the single image extract the 4 patches
        # with input shape 720x1280 each patch has shape 240x320
        patches = tf.image.extract_patches(
            images=image,
            sizes=[1, 240, 320, 1],
            strides=[1, 240, 320, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        patches = tf.reshape(patches, (12, 240, 320, 3))

        return patches

    # extract patches
    trainval_dataX = trainval_dataX.map(_extract_patches, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    trainval_dataY = trainval_dataY.map(_extract_patches, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    test_dataX = test_dataX.map(_extract_patches, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataY = test_dataY.map(_extract_patches, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # now each element of the dataset has shape (12, 240, 320, 3)
    # un-batch in order to have each element of shape (1, 240, 320, 3)
    trainval_dataX = trainval_dataX.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    trainval_dataY = trainval_dataY.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))

    test_dataX = test_dataX.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    test_dataY = test_dataY.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))

    # Zip again
    trainval_data = tf.data.Dataset.zip((trainval_dataX, trainval_dataY))
    test_data = tf.data.Dataset.zip((test_dataX, test_dataY))

    # Shuffle once and perform train-validation split
    trainval_data = trainval_data.shuffle(buffer_size=50, seed=seed, reshuffle_each_iteration=False)
    train_data = trainval_data.skip(val_size)
    val_data = trainval_data.take(val_size)

    def _random_flip(image_blur, image_sharp):
        do_flip = tf.random.uniform([], seed=seed) > 0.5
        image_blur = tf.cond(do_flip, lambda: tf.image.flip_left_right(image_blur), lambda: image_blur)
        image_sharp = tf.cond(do_flip, lambda: tf.image.flip_left_right(image_sharp), lambda: image_sharp)

        return image_blur, image_sharp

    # Perform augmentation on train set only
    train_data = train_data.map(_random_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Reshuffle train and validation sets each epoch
    train_data = train_data.shuffle(buffer_size=BUF, seed=seed, reshuffle_each_iteration=True)
    val_data = val_data.shuffle(buffer_size=BUF, seed=seed, reshuffle_each_iteration=True)

    # Cache, batch and repeat train and validation sets
    train_data = train_data.cache().batch(batch_size).repeat(epochs)
    val_data = val_data.cache().batch(batch_size).repeat(epochs)

    # Batch test set
    test_data = test_data.batch(batch_size)

    # Prefetch
    train_data.prefetch(tf.data.experimental.AUTOTUNE)
    val_data.prefetch(tf.data.experimental.AUTOTUNE)

    return train_data, test_data, val_data
