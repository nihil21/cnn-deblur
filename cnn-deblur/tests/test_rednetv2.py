from models.rednet import REDNetV2
from datasets import cifar_dataset
import tensorflow as tf

rednet = REDNetV2((32, 32, 3))
rednet.summary()
rednet.plot_model('test.png')

train, _, _ = cifar_dataset.load_image_dataset()
data = tf.data.Dataset.from_tensor_slices(train)
data = data.batch(32)
for d in data:
    print(d[0].shape)
    break


def split(blurred, sharp):
    blurred_ch = tf.split(blurred, num_or_size_splits=3, axis=3)
    sharp_ch = tf.split(sharp, num_or_size_splits=3, axis=3)
    return blurred_ch, sharp_ch


data = data.map(split)
for d in data:
    blur_ch = d[0]
    print(len(blur_ch))
    for ch in blur_ch:
        print(ch.shape)
    break
