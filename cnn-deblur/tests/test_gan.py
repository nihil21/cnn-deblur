from models.deblur_gan import DeblurGan
from datasets import cifar_dataset

deblur_gan = DeblurGan(input_shape=(32, 32, 3))
train_data, _ = cifar_dataset.load_image_dataset(do_val_split=False, normalize=True)
deblur_gan.train((train_data[0][0:1], train_data[1][0:1]),
                 epochs=1,
                 batch_size=2)
