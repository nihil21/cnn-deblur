from utils.preproc_cifar import *


def main():
    # Preprocess Cifar10 dataset
    (trainX, trainY), (testX, testY) = preproc_cifar()

    print('Train set size: {0:d}'.format(trainX.shape[0]))
    print('Test set size: {0:d}'.format(testX.shape[0]))

    # Chose random images
    train_idx = np.random.randint(0, trainX.shape[0])
    test_idx = np.random.randint(0, testX.shape[0])

    # Prepare sample results for displaying
    sample_trainX = cv2.cvtColor(trainX[train_idx], cv2.COLOR_BGR2RGB)
    sample_trainY = cv2.cvtColor(trainY[train_idx], cv2.COLOR_BGR2RGB)

    sample_testX = cv2.cvtColor(testX[test_idx], cv2.COLOR_BGR2RGB)
    sample_testY = cv2.cvtColor(testY[test_idx], cv2.COLOR_BGR2RGB)

    sample_trainX = cv2.resize(sample_trainX, (300, 300), interpolation=cv2.INTER_LINEAR)
    sample_trainY = cv2.resize(sample_trainY, (300, 300), interpolation=cv2.INTER_LINEAR)

    sample_testX = cv2.resize(sample_testX, (300, 300), interpolation=cv2.INTER_LINEAR)
    sample_testY = cv2.resize(sample_testY, (300, 300), interpolation=cv2.INTER_LINEAR)

    # Display results
    sample_train = np.hstack((sample_trainX, sample_trainY))
    sample_test = np.hstack((sample_testX, sample_testY))
    cv2.imshow('Sample train [{:d}]'.format(train_idx), sample_train)
    cv2.imshow('Sample test [{:d}]'.format(test_idx), sample_test)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
