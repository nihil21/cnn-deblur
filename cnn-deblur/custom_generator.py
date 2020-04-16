import tensorflow as tf
import matplotlib.pyplot as plt
import time


def load_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image)
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image


def resize_image(image, new_height, new_width):
    return tf.image.resize(image, [new_height, new_width])


def random_flip(image_blur, image_sharp, seed):
    do_flip = tf.random.uniform([], seed=seed) > 0.5
    image_blur = tf.cond(do_flip, lambda: tf.image.flip_left_right(image_blur), lambda: image_blur)
    image_sharp = tf.cond(do_flip, lambda: tf.image.flip_left_right(image_sharp), lambda: image_sharp)

    return image_blur, image_sharp


# DEBUG
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.axis('off')


# DEBUG
def show_batch(batch):
    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 2
    for i in range(1, columns * rows + 1):
        img = batch[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


start_time = time.time()
seed = 42
epochs = 10
batch_size = 4
new_dimension = [720, 1280]
total_elements = 24000
validation_split = 0.125

blur = tf.data.Dataset.list_files('/media/tree/Dati/GIACOMO/Unibo/Magistrale/Ianno/Deep Learning/Project/Tests/dataset/blur/train/*/*', shuffle=True, seed=seed)
sharp = tf.data.Dataset.list_files('/media/tree/Dati/GIACOMO/Unibo/Magistrale/Ianno/Deep Learning/Project/Tests/dataset/sharp/train/*/*', shuffle=True, seed=seed)

blur = blur.map(lambda filename: load_image(filename))
sharp = sharp.map(lambda filename: load_image(filename))

blur = blur.map(lambda image: resize_image(image, new_dimension[0], new_dimension[1]))
sharp = sharp.map(lambda image: resize_image(image, new_dimension[0], new_dimension[1]))

dataset = tf.data.Dataset.zip((blur, sharp))
# reshuffle_each_iteration=False ensures that train and validation set are disjoint
dataset = dataset.shuffle(buffer_size=1000, seed=seed, reshuffle_each_iteration=False)

# train and validation split
train = dataset.skip(int(total_elements*validation_split))
validation = dataset.take(int(total_elements*validation_split))

train_augmented = train.map(lambda image_blur, image_sharp: random_flip(image_blur, image_sharp, seed))
train_augmented = train_augmented.shuffle(buffer_size=1000, seed=42, reshuffle_each_iteration=True)

# repeat: once for each epoch
train_augmented = train_augmented.batch(batch_size).repeat(epochs)
validation = validation.batch(batch_size).repeat(epochs)

train_augmented.prefetch(10)
validation.prefetch(10)
print('Time elapsed: {0:.2f} s'.format(time.time() - start_time))

"""
# DEBUG
def flip (x, y, seed):
    do_flip = tf.random.uniform([], seed=seed) > 0.5
    x = tf.cond(do_flip, lambda: -x, lambda: x)
    return x, y

seed = 42
total_elements = 100

dataset = tf.data.Dataset.from_tensor_slices(list(range(total_elements)))
dataset = dataset.shuffle(buffer_size=1000, seed=42, reshuffle_each_iteration=False)

train_dataset = dataset.skip(int(total_elements*validation_split))
val_dataset = dataset.take(int(total_elements*validation_split))

train_dataset = train_dataset.shuffle(buffer_size=1000, seed=42, reshuffle_each_iteration=True)

batched_train_dataset = train_dataset.batch(batch_size).repeat(epochs)
batched_val_dataset = val_dataset.batch(batch_size).repeat(epochs)

for b in batched_train_dataset:
    print(b)

print("\n\n")

for b in batched_val_dataset:
    print(b)
"""
