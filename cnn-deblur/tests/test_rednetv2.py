from models.rednet import REDNetV2
from datasets import cifar_dataset
import tensorflow as tf

rednet = REDNetV2((32, 32, 3),  num_layers=2)
rednet.build((None, 32, 32, 3))
rednet.summary()
# rednet.plot_model('test.png')

train, _, _ = cifar_dataset.load_image_dataset()
train = train[:5]
data = tf.data.Dataset.from_tensor_slices(train)
data = data.batch(2)
for d in data:
    print(d[0].shape)
    break


def split(b, s):
    # blurred_ch = tf.split(blurred, num_or_size_splits=3, axis=3)
    s_ch = tf.split(s, num_or_size_splits=3, axis=3)
    return b, s_ch


data = data.map(split)
for d in data:
    (blurred, sharp) = d
    print(f'Blurred shape: {blurred.shape}')
    print(f'Sharp len: {len(sharp)}, sharp shape: {sharp[0].shape}')
    restored = rednet.model(blurred)
    print(f'Restored len: {len(restored)}, restored shape: {restored[0].shape}')
    break
