import numpy as np
import os


# ----------------------------------------------------------------------------------------------------------------------
#               5.1 DATASET READER
# ----------------------------------------------------------------------------------------------------------------------
def dataset_reader(file_path):
    """
    Reads the input vectors and the corresponding labels from a file
    @param file_path: String / path to files
    @return: two numpy arrays / 1. input vectors, 2. labels
    """
    vectors = []
    labels = []
    with open(file_path, "r") as f:
        for line in f:
            text, label, str_vector = line.split('\t')
            # Convert labels to corresponding int values
            if label.endswith('POS'):
                labels.append(1)
            elif label.endswith('NEG'):
                labels.append(0)
            # Convert string values to float and add bias
            vector = np.append(np.array(str_vector.split(), dtype='float'), np.ones(1))
            vectors.append(vector)

    return np.array(vectors), np.array(labels)


# ----------------------------------------------------------------------------------------------------------------------
#               5.2 NUMPY IMPLEMENTATION
# ----------------------------------------------------------------------------------------------------------------------
def sigmoid(x):
    """
    Implements the sigmoid activation function
    @param x: float
    @return: float
    """
    return 1.0 / (1 + np.exp(-x))


def loss(X, y, w):
    """
    Computes mean squared loss
    @param X: numpy array / input vectors
    @param y: numpy array / desired outputs
    @param w: numpy array / perceptron weight
    @return: float / squared loss
    """
    return np.average(np.square(sigmoid(X @ w) - y))


def accuracy(X, y, w):
    """
    Computes accuracy
    @param X: numpy array / input vectors
    @param y: numpy array / desired outputs
    @param w: numpy array / perceptron weight
    @return: float / accuracy
    """
    # Use sigmoid activation function and round results to nearest integer
    predictions = np.rint(sigmoid(X @ w))
    return np.sum(predictions == y) / len(y)


def perceptron_train(train, labels, w, epochs, batch_size, learning_rate):
    """
    Implements the training of the perceptron
    @param train: numpy array / input vectors of the train set
    @param labels: numpy array / labels of the train set
    @param w: numpy array / weights of the perceptron
    @param epochs: int / number of epochs
    @param batch_size: int
    @param learning_rate: int
    @return: numpy array / learned weights of the perceptron
    """
    for i in range(epochs):
        # get shuffling indices
        shuffling = np.random.permutation(labels.shape[0])
        # iterate over dataset in batches
        for batch in np.array_split(shuffling, np.ceil(len(train) / batch_size)):
            input_vecs = train[batch]
            des_output = labels[batch]
            gradient = 0
            # Implement weight update
            for x, y in zip(input_vecs, des_output):
                sig_xw = sigmoid(x @ w)
                gradient += (sig_xw - y) * sig_xw * (1 - sig_xw) * x
            w -= (learning_rate / batch_size) * gradient
        # Track train loss
        if i % 25 == 0:
            print("Epoch: ", i, ", Train loss: ", loss(train, labels, w))
    return w


# ----------------------------------------------------------------------------------------------------------------------
#               5.3 TRAINING
# ----------------------------------------------------------------------------------------------------------------------
np.random.seed(42)
# Load data
X_train, y_train = dataset_reader('D:\DL4NLP\Homework\hw01_data\DATA/rt-polarity.train.vecs')
X_dev, y_dev = dataset_reader('D:\DL4NLP\Homework\hw01_data\DATA/rt-polarity.dev.vecs')
X_test, y_test = dataset_reader('D:\DL4NLP\Homework\hw01_data\DATA/rt-polarity.test.vecs')

print("First run with given hyperparameters...")
trained_weights = perceptron_train(X_train, y_train, np.random.normal(0, 1, 101), 50, 10, 0.01)
# Loss and accuracy on test set
print("Dev loss: ", loss(X_dev, y_dev, trained_weights))
print("Dev accuracy: ", accuracy(X_dev, y_dev, trained_weights))

print("Second run with optimized hyperparameters...")
better_weights = perceptron_train(X_train, y_train, np.random.normal(0, 1, 101), 250, 25, 0.03)
# Loss and accuracy on test set
print("Dev loss: ", loss(X_dev, y_dev, better_weights))
print("Dev accuracy: ", accuracy(X_dev, y_dev, better_weights))
print("Test loss: ", loss(X_test, y_test, better_weights))
print("Test accuracy: ", accuracy(X_test, y_test, better_weights))