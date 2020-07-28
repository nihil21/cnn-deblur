from utils.custom_losses import perceptual_loss
from datasets import cifar_dataset

train_data, _ = cifar_dataset.load_image_dataset(do_val_split=False, normalize=True)
loss = perceptual_loss(train_data[0][0:4], train_data[1][0:4])
print(loss)
