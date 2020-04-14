import tensorflow as tf
import matplotlib.pyplot as plt

def load_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image)
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image

def resize_image(image, new_height, new_width):
    return tf.image.resize(image, [new_height, new_width])


# DEBUG
def show(image):
  plt.figure()
  plt.imshow(image)
  plt.axis('off')

def show_batch(batch):
    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 2
    for i in range(1, columns * rows + 1):
        img = batch[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()

epochs = 10
batch_size = 4
new_dimension = [1280, 720]

blur = tf.data.Dataset.list_files('/media/tree/Dati/GIACOMO/Unibo/Magistrale/Ianno/Deep Learning/Project/Tests/dataset/blur/train/*/*', shuffle=True, seed=42)
sharp = tf.data.Dataset.list_files('/media/tree/Dati/GIACOMO/Unibo/Magistrale/Ianno/Deep Learning/Project/Tests/dataset/sharp/train/*/*', shuffle=True, seed=42)

blur = blur.map(lambda filename: load_image(filename))
sharp = sharp.map(lambda filename: load_image(filename))

blur = blur.map(lambda image: resize_image(image, new_dimension[0], new_dimension[1]))
sharp = sharp.map(lambda image: resize_image(image, new_dimension[0], new_dimension[1]))

dataset = tf.data.Dataset.zip((blur, sharp))

dataset = dataset.shuffle(buffer_size=1000, seed=42)

# repeat: once for each epoch
batched_dataset = dataset.batch(batch_size).repeat(epochs)

for batch in batched_dataset.take(1):
    print(batch[0].shape)

    show_batch(batch[0])
    show_batch(batch[1])

# TODO: image mirror