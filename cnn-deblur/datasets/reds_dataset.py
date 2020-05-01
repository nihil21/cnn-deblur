import tensorflow as tf
import os
from glob import glob


def load_image_dataset(dataset_root,
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

    def _extract_patches(img):
        img = tf.reshape(img, (1, 720, 1280, 3))
        # from the single image extract the 4 patches corresponding to the 4 corners
        # with input shape 720x1280 each patch has shape 360x640
        patches = tf.image.extract_patches(images=img,
                                           sizes=[1, 360, 640, 1],
                                           strides=[1, 360, 640, 1],
                                           rates=[1, 1, 1, 1],
                                           padding='VALID')

        patches = tf.reshape(patches, (4, 360, 640, 3))

        return patches

    # extract patches
    blur = blur.map(_extract_patches, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # now each element of the dataset has shape (4, 360, 640, 3)
    # un-batch in order to have each element of shape (1, 360, 640, 3)
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
    blur_test = blur_test.map(_break_image)
    blur_test = blur_test.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    sharp_test = sharp_test.map(_break_image)
    sharp_test = sharp_test.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    test = tf.data.Dataset.zip((blur_test, sharp_test))
    test = test.batch(batch_size)

    return train_augmented, test, validation


def load_tfrecord_dataset(dataset_root,
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
    trainval_data = trainval_data.map(_parse_image_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_data = test_data.map(_parse_image_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def _extract_patches(image_blur, image_sharp):
        image_blur = tf.reshape(image_blur, (1, 720, 1280, 3))
        image_sharp = tf.reshape(image_sharp, (1, 720, 1280, 3))
        # from the single image extract the 4 patches corresponding to the 4 corners
        # with input shape 720x1280 each patch has shape 360x640
        patches_blur = tf.image.extract_patches(images=image_blur,
                                                sizes=[1, 360, 640, 1],
                                                strides=[1, 360, 640, 1],
                                                rates=[1, 1, 1, 1],
                                                padding='VALID')

        patches_sharp = tf.image.extract_patches(images=image_sharp,
                                                 sizes=[1, 360, 640, 1],
                                                 strides=[1, 360, 640, 1],
                                                 rates=[1, 1, 1, 1],
                                                 padding='VALID')

        patches_blur = tf.reshape(patches_blur, (4, 360, 640, 3))
        patches_sharp = tf.reshape(patches_sharp, (4, 360, 640, 3))

        return patches_blur, patches_sharp

    # extract patches
    trainval_data = trainval_data.map(_extract_patches, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_data = test_data.map(_extract_patches, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # now each element of the dataset has shape (4, 360, 640, 3)
    # un-batch in order to have each element of shape (1, 360, 640, 3)
    trainval_data = trainval_data.flat_map(lambda x, y: (tf.data.Dataset.from_tensor_slices(x), tf.data.Dataset.from_tensor_slices(y)))
    test_data = test_data.flat_map(lambda x, y: (tf.data.Dataset.from_tensor_slices(x), tf.data.Dataset.from_tensor_slices(y)))

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
