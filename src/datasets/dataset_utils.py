import os
import typing

import tensorflow as tf
from google.cloud import storage


def load_dataset_from_gcs(
        project_id: str,
        bucket_name: str,
        prefix: str,
        res: typing.Tuple[int, int],
        val_size: int,
        batch_size: int,
        epochs: int,
        seed: int = 42,
        use_patches: bool = False,
        repeat: bool = True,
        zero_mean: bool = False
):
    # Shuffle buffer size
    BUF = 50

    # Connect to bucket
    client = storage.Client(project_id)
    bucket = client.bucket(bucket_name)

    # Load .tfrecords files
    tf_trainval = [
        os.path.join('gs://{0:s}'.format(bucket_name), f.name)
        for f in bucket.list_blobs(prefix='{0:s}/train'.format(prefix))
    ]
    trainval_data = tf.data.TFRecordDataset(
        filenames=tf_trainval,
        num_parallel_reads=tf.data.experimental.AUTOTUNE
    )
    tf_test = [
        os.path.join('gs://{0:s}'.format(bucket_name), f.name)
        for f in bucket.list_blobs(prefix='{0:s}/test'.format(prefix))
    ]
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
        blur_img = tf.image.resize(blur_img, res)
        sharp_img = tf.image.decode_png(images['sharp'], channels=3)
        sharp_img = tf.image.resize(sharp_img, res)

        blur_img /= 255.0
        sharp_img /= 255.0

        if zero_mean:
            blur_img = 2.0 * blur_img - 1.0
            sharp_img = 2.0 * sharp_img - 1.0

        blur_img = tf.cast(blur_img, dtype=tf.bfloat16)
        sharp_img = tf.cast(sharp_img, dtype=tf.bfloat16)

        return blur_img, sharp_img

    # Map parsing function
    trainval_data = trainval_data.map(
        _parse_image_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    test_data = test_data.map(
        _parse_image_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # Shuffle once and perform train-validation split
    trainval_data = trainval_data.shuffle(buffer_size=50, seed=seed, reshuffle_each_iteration=False)
    train_data = trainval_data.skip(val_size)
    val_data = trainval_data.take(val_size)

    if use_patches:
        train_data = extract_patches_from_dataset(train_data)
        val_data = extract_patches_from_dataset(val_data)

    # Cache training and validation sets
    train_data = train_data.cache()
    val_data = val_data.cache()

    def _random_horizontal_flip(image_blur, image_sharp):
        do_flip = tf.random.uniform([], seed=seed) > 0.5
        image_blur = tf.cond(do_flip, lambda: tf.image.flip_left_right(image_blur), lambda: image_blur)
        image_sharp = tf.cond(do_flip, lambda: tf.image.flip_left_right(image_sharp), lambda: image_sharp)

        return image_blur, image_sharp

    def _random_vertical_flip(image_blur, image_sharp):
        do_flip = tf.random.uniform([], seed=seed) > 0.5
        image_blur = tf.cond(do_flip, lambda: tf.image.flip_up_down(image_blur), lambda: image_blur)
        image_sharp = tf.cond(do_flip, lambda: tf.image.flip_up_down(image_sharp), lambda: image_sharp)

        return image_blur, image_sharp

    # Perform augmentation on training set
    train_data = train_data.map(_random_horizontal_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_data = train_data.map(_random_vertical_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Reshuffle training and validation sets
    train_data = train_data.shuffle(buffer_size=BUF, seed=seed, reshuffle_each_iteration=True)
    val_data = val_data.shuffle(buffer_size=BUF, seed=seed, reshuffle_each_iteration=True)

    # Batch train and validation sets
    train_data = train_data.batch(batch_size)
    val_data = val_data.batch(batch_size)

    # Repeat train and validation sets if required
    if repeat:
        train_data = train_data.repeat(epochs)
        val_data = val_data.repeat(epochs)

    # Batch test set only if use_patches is set to false
    if not use_patches:
        test_data = test_data.batch(batch_size)

    # Prefetch
    train_data.prefetch(tf.data.experimental.AUTOTUNE)
    val_data.prefetch(tf.data.experimental.AUTOTUNE)

    return train_data, test_data, val_data


def extract_patches_from_dataset(dataset):
    # Unzip dataset
    datasetX = dataset.map(lambda x, y: x)
    datasetY = dataset.map(lambda x, y: y)

    # Extract patches
    datasetX = datasetX.map(extract_patches, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    datasetY = datasetY.map(extract_patches, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Now each element of the dataset has shape (12, 96, 128, 3);
    # un-batch in order to have each element of shape (1, 96, 128, 3)
    datasetX = datasetX.flat_map(tf.data.Dataset.from_tensor_slices)
    datasetY = datasetY.flat_map(tf.data.Dataset.from_tensor_slices)

    # Zip again
    return tf.data.Dataset.zip((datasetX, datasetY))


def extract_patches(image):
    res = image.shape[0:2]
    # From the single image extract 12 patches (3x4)
    # e.g. with input shape 288x512 each patch has shape 96x128
    patches = tf.image.extract_patches(
        images=tf.expand_dims(image, 0),
        sizes=[1, res[0] // 3, res[1] // 4, 1],
        strides=[1, res[0] // 3, res[1] // 4, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )

    patches = tf.reshape(patches, (12, res[0] // 3, res[1] // 4, 3))

    return patches


def reconstruct_image(image_patches, patch_size: typing.Tuple[int, int]):
    restored_image = []
    for i in range(0, patch_size[0]):
        row = []
        for j in range(0, patch_size[1]):
            row.append(image_patches[i * 4 + j])

        restored_row = tf.concat(row, axis=1)
        restored_image.append(restored_row)

    return tf.concat(restored_image, axis=0)
