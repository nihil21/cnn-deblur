import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model.conv_net import ConvNet
from model.u_net import UNet

conv_net = UNetREDS(input_shape=(1280, 720, 3))
conv_net.compile(loss='mse')

bs = 64
seed = 42

trainX_datagen = ImageDataGenerator(horizontal_flip=True)
trainY_datagen = ImageDataGenerator(horizontal_flip=True)

validX_datagen = ImageDataGenerator()
validY_datagen = ImageDataGenerator()

trainX_flow = trainX_datagen.flow_from_directory(directory='/home/uni/dataset/train/train_blur/',
                                                target_size=(1280, 720),
                                                color_mode='rgb',
                                                batch_size=bs,
                                                class_mode=None,
                                                shuffle=True,
                                                seed=seed)
trainY_flow = trainY_datagen.flow_from_directory(directory='/home/uni/dataset/train/train_sharp/',
                                                target_size=(1280, 720),
                                                color_mode='rgb',
                                                batch_size=bs,
                                                class_mode=None,
                                                shuffle=True,
                                                seed=seed)


validX_flow = validX_datagen.flow_from_directory(directory='/home/uni/dataset/val/val_blur/',
                                                target_size=(1280, 720),
                                                color_mode='rgb',
                                                batch_size=bs,
                                                class_mode=None,
                                                shuffle=True,
                                                seed=seed)
validY_flow = validY_datagen.flow_from_directory(directory='/home/uni/dataset/val/val_sharp/',
                                                target_size=(1280, 720),
                                                color_mode='rgb',
                                                batch_size=bs,
                                                class_mode=None,
                                                shuffle=True,
                                                seed=seed)


STEP_SIZE_TRAIN=trainX_flow.n//trainX_flow.batch_size
STEP_SIZE_VALID=validX_flow.n//validX_flow.batch_size

train_data = (pair for pair in zip(trainX_flow, trainY_flow))
valid_data = (pair for pair in zip(validX_flow, validY_flow))

conv_net.fit(train_data,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_data,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10
)