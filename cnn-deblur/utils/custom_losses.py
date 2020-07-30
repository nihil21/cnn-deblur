import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from typing import Optional


def wasserstein_loss(trueY, predY):
    return K.mean(trueY * predY)


def perceptual_loss(trueY, predY, loss_model: Optional[Model] = None):
    if loss_model is None:
        vgg = VGG16(include_top=False, weights='imagenet', input_shape=(None, None, 3))
        loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
        loss_model.trainable = False
    return K.mean(K.square(loss_model(trueY) - loss_model(predY)))
