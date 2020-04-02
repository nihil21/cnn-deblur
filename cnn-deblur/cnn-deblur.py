from utils.preproc_cifar import *
from model.conv_net import ConvNet
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    print(tf.config.experimental.list_physical_devices('GPU'))

    # Preprocess Cifar10 dataset
    (trainX, trainY), (testX, testY) = preproc_cifar(normalize=True)

    # Reserve samples for validation
    valX = trainX[-10000:]
    valY = trainY[-10000:]

    print('Training set size: {0:d}'.format(trainX.shape[0]))
    print('Validation set size: {0:d}'.format(valX.shape[0]))
    print('Test set size: {0:d}'.format(testX.shape[0]))

    # Create ResNet and plot model
    conv_net = ConvNet(input_shape=(32, 32, 3))
    conv_net.plot_model(os.path.join('..', 'res', 'model.png'))

    # Train model
    conv_net.fit(trainX, trainY,
                 batch_size=64,
                 epochs=5,
                 validation_data=(valX, valY))

    # Evaluate the model on the test data
    results = conv_net.model.evaluate(testX, testY, batch_size=128)
    print('test loss, test acc:', results)

    # Generate predictions on new data
    predictions = conv_net.predict(testX[:3])
    idx = 0
    for img in predictions:
        img = cv2.normalize(img,
                            dst=None,
                            alpha=0,
                            beta=255,
                            norm_type=cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join('..', 'res', 'Out{0:d}.jpg'.format(idx)), img)
        idx += 1


if __name__ == '__main__':
    main()
