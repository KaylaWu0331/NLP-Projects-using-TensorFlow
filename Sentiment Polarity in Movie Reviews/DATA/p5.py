import numpy as np
import os

# ----------------------------------------------------------------------------------------------------------------------
#               5.1 DATASET READER

# ----------------------------------------------------------------------------------------------------------------------
def data_reader(file):
    with open(file, encoding='utf-8') as f:
        lines = f.readlines()
    veclist = []
    labellist = []
    for line in lines:
        # Encode the corresponding label as y = 1 for POS and y = 0 for NEG
        str1 = line.split('\t')[1]
        if str1 == 'label=POS':
            labellist.append(1)
        elif str1 == 'label=NEG':
            labellist.append(0)
        else:
            break
        # the third entry is a 100-dimensional vector representing the review
        list1 = line.split('\t')[2].split()
        list2 = []
        for i in list1:
            list2.append(float(i))
        # Add a bias, i.e., append a trailing 1 to each input vector x.
        list2.append(float(1))
        arr1 = np.array(list2)
        veclist.append(arr1)

    return veclist, labellist

veclist_dev,labellist_dev = data_reader('DATA/rt-polarity.dev.vecs')
veclist_test,labellist_test = data_reader('DATA/rt-polarity.test.vecs')
veclist_train,labellist_train = data_reader('DATA/rt-polarity.train.vecs')

# ----------------------------------------------------------------------------------------------------------------------
#               5.2 NUMPY IMPLEMENTATION
# ----------------------------------------------------------------------------------------------------------------------
# activation function
def Sigmoid(Z):
    #print(Z)
    #print(1 / (1 + np.exp(-0.005 * Z)))
    #print('----------')
    return 1 / (1 + np.exp(-0.005 * Z))

# Derivatives of sigmoid functions
def Derivative_sigmoid(Z):
    return Sigmoid(Z) * (1 - Sigmoid(Z))

# Loss function
def Loss(X, y, w):
    loss = 0
    for i in range(len(X)):
        #print(Sigmoid(np.dot(X[i], w)) - y[i])
        loss += np.square(Sigmoid(np.dot(X[i], w)) - y[i])
    return loss


#implementing random mini-batches is to randomly shuffle the whole training dataset
def mini_batch(mini_batch_size, X, y, seed):
    np.random.seed(seed)
    mini_batches = []
    # randomly shuffle the whole training dataset
    shuffled_X = np.random.permutation(X)
    shuffled_y = np.random.permutation(y)
    # divide the training dataset into batches of size |T ′|
    num_of_minibatch = int(len(X) / mini_batch_size)

    for k in range(num_of_minibatch):
        mini_batch_X = shuffled_X[k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_y = shuffled_y[k * mini_batch_size:(k + 1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_y)
        mini_batches.append(mini_batch)
    # If there is not enough left for a batch size, use the rest as a batch
    if len(X) % mini_batch_size != 0:
        mini_batch_X = shuffled_X[mini_batch_size * num_of_minibatch:]
        mini_batch_y = shuffled_y[mini_batch_size * num_of_minibatch:]

        mini_batch = (mini_batch_X, mini_batch_y)
        mini_batches.append(mini_batch)

    #return a random subset of the whole training data T .
    return mini_batches[np.random.randint(0, num_of_minibatch)]


# For the weight update rule, use the mini-batch stochastic gradient descent formula:
def Weight_update(lr, epochs, mini_batch_size, seed, w):
    #decay = lr / epochs
    for i in range(epochs):
        seed = seed + 1
        (mini_batch_X, mini_batch_y) = mini_batch(mini_batch_size, veclist_train, labellist_train, seed)
        #calculate the sum part
        sum1 = np.zeros(101)
        for j in range(len(mini_batch_X)):
            #print(mini_batch_X[j],w)
            sum1 += (Sigmoid(np.dot(mini_batch_X[j], w)) - mini_batch_y[j]) * Derivative_sigmoid(
                np.dot(mini_batch_X[j], w)) * mini_batch_X[j]

        w = w - (lr / mini_batch_size) * sum1
        #Loss
        #print('lr:',lr)
        print('train loss after epoch', i+1, ' : ', Loss(veclist_train, labellist_train, w))
        print('test loss:',Loss(veclist_dev,labellist_dev,w))

       # lr = lr / (1 + decay * (i+1))

    return w

# predict function
def Predict_fun(X, w):
    predictlist = []
    for i in range(len(X)):
        res = Sigmoid(np.dot(X[i], w))
        #print(res)
        if res >= 0.5:
            predictlist.append(1)
            #print('pos')
        else:
            predictlist.append(0)
            #print('neg')
    return np.array(predictlist)
# compute the accuracy
def Cal_accuracy(label, predict):
    i = 0
    for j in range(len(label)):
        #Count the number of correct predictions
        if label[j] == predict[j]:
            i += 1
        else:
            pass
    return i / len(label)

# ----------------------------------------------------------------------------------------------------------------------
#               5.3 TRAINING
# ----------------------------------------------------------------------------------------------------------------------
mini_batch_size = 50
lr = 0.3
epochs = 300
seed = 0

#Initialize the weight vector (100-dimensional vector + 1-bias)
w = np.random.normal(0,1,(101,))
#print('The initial w', w)

#Train perceptron on the training data
w_new = Weight_update(lr,epochs,mini_batch_size,seed,w)

#accuracy on the dev set
print('result on dev set:')
print('Accuracy:', Cal_accuracy(labellist_dev,Predict_fun(veclist_dev,w_new)))
print('loss:', Loss(veclist_dev,labellist_dev,w_new))

#accuracy on the test set
print('result on test set:')
print('Accuracy:', Cal_accuracy(labellist_test,Predict_fun(veclist_test,w_new)))
print('loss:', Loss(veclist_test,labellist_test,w_new))