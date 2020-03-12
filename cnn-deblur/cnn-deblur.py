from utils.preproc_cifar import *


def main():
    # Preprocess Cifar10 dataset
    (trainX, trainY), (testX, testY) = preproc_cifar()

    # Prepare sample results for displaying
    sample_train_noise = cv2.cvtColor(trainX[5], cv2.COLOR_BGR2RGB)
    sample_train_clear = cv2.cvtColor(trainY[5], cv2.COLOR_BGR2RGB)

    sample_test_noise = cv2.cvtColor(testX[5], cv2.COLOR_BGR2RGB)
    sample_test_clear = cv2.cvtColor(testY[5], cv2.COLOR_BGR2RGB)

    sample_train_noise = cv2.resize(sample_train_noise, (300, 300), interpolation=cv2.INTER_LINEAR)
    sample_train_clear = cv2.resize(sample_train_clear, (300, 300), interpolation=cv2.INTER_LINEAR)

    sample_test_noise = cv2.resize(sample_test_noise, (300, 300), interpolation=cv2.INTER_LINEAR)
    sample_test_clear = cv2.resize(sample_test_clear, (300, 300), interpolation=cv2.INTER_LINEAR)

    # Display results
    sample_train = np.hstack((sample_train_noise, sample_train_clear))
    sample_test = np.hstack((sample_test_noise, sample_test_clear))
    cv2.imshow('Sample train', sample_train)
    cv2.imshow('Sample test', sample_test)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
