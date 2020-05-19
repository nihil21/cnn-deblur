import tensorflow as tf
from model.custom_losses_metrics import ssim, psnr_loss
from datasets import cifar_dataset
import matplotlib.pyplot as plt


train_data, _, _ = cifar_dataset.load_image_dataset(normalize=True)
img1 = tf.convert_to_tensor(train_data[1][0], dtype=tf.float32)
img2 = tf.convert_to_tensor(train_data[1][0], dtype=tf.float32)

_, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(img1)
ax2.imshow(img2)
plt.show()

print('SSIM metric (should be 1):', ssim(img1, img2))
print('PSNR (should be -inf):', psnr_loss(img1, img2))
print('-'*50)

img2 = tf.ones(shape=(32, 32, 3)) - img1

_, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(img1)
ax2.imshow(img2)
plt.show()

print('SSIM metric (should be 0):', ssim(img1, img2))
print('PSNR (should be 0):', psnr_loss(img1, img2))


print('-'*50)
img1 = tf.convert_to_tensor(train_data[0][0])
img2 = tf.convert_to_tensor(train_data[1][0])

_, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(img1)
ax2.imshow(img2)
plt.show()

print('SSIM metric:', ssim(img1, img2))
print('PSNR:', psnr_loss(img1, img2))
