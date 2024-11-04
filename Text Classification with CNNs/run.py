import numpy as np
import random
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *

# ------------------------------------------------
#             2.1 Creating Data Splits
# ------------------------------------------------
################################
input_file = 'data.txt'
################################

tmp_dir = '/tmp'
train_verbose = 1
pad_length = 300


def read_data(input_file):
    vocab = {0}
    data_x = []
    data_y = []
    with open(input_file) as f:
        for line in f:
            label, content = line.split('\t')
            content = [int(v) for v in content.split()]
            vocab.update(content)
            data_x.append(content)
            label = tuple(int(v) for v in label.split())
            data_y.append(label)

    data_x = pad_sequences(data_x, maxlen=pad_length)
    return list(zip(data_y, data_x)), vocab


data, vocab = read_data(input_file)
vocab_size = max(vocab) + 1

# random seeds
random.seed(42)
tf.random.set_seed(42)

random.shuffle(data)
input_len = len(data)

num_classes = 20

# train_y: a list of 20-component one-hot vectors representing newsgroups
# train_y: a list of 300-component vectors where each entry corresponds to a word ID
train_y, train_x = zip(*(data[:(input_len * 8) // 10]))
dev_y, dev_x = zip(*(data[(input_len * 8) // 10: (input_len * 9) // 10]))
test_y, test_x = zip(*(data[(input_len * 9) // 10:]))

# ------------------------------------------------
#                 2.2 A Basic CNN
# ------------------------------------------------

train_x, train_y = np.array(train_x), np.array(train_y)
dev_x, dev_y = np.array(dev_x), np.array(dev_y)
test_x, test_y = np.array(test_x), np.array(test_y)

# Leave those unmodified and, if requested by the task, modify them locally in the specific task
batch_size = 64
embedding_dims = 100
epochs = 2
filters = 75
kernel_size = 3  # Keras uses a different definition where a kernel size of 3 means that 3 words are convolved at each step

model = Sequential()
model.add(Embedding(vocab_size, embedding_dims, input_length=pad_length))

####################################
#                                  #
#   add your implementation here   #
model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(num_classes))
model.add(Activation('softmax'))

print(model.summary())
#                                  #
####################################

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=train_verbose)
print('Accuracy of simple CNN: %f\n' % model.evaluate(dev_x, dev_y, verbose=0)[1])

# console output:
#    Accuracy of simple CNN: 0.766861


# ------------------------------------------------
#                2.3 Early Stopping
# ------------------------------------------------


####################################
#                                  #
#   add your implementation here   #
from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(patience=2, monitor='val_accuracy', verbose=1)
mc_path = os.path.join(tmp_dir, 'model-task2.3.hdf5')
mc = ModelCheckpoint(filepath=mc_path, monitor='val_accuracy', save_best_only=True, save_weights_only=True, verbose=1)

model.reset_states()
history = model.fit(train_x, train_y, batch_size=batch_size, epochs=50, verbose=train_verbose,
                    validation_data=(dev_x, dev_y), callbacks=[es, mc])
model.load_weights(mc_path)

print('Accuracy (dev) of CNN: %f' % model.evaluate(dev_x, dev_y, verbose=0)[1])
print('Accuracy (test) of CNN: %f' % model.evaluate(test_x, test_y, verbose=0)[1])
#                                  #
####################################

# console output:
#   Train on 15062 samples, validate on 1883 samples
#   Epoch 00001: val_acc improved from -inf to 0.64153, saving model to /tmp/model-task2.3.hdf5
#   Epoch 00002: val_acc improved from 0.64153 to 0.77270, saving model to /tmp/model-task2.3.hdf5
#   Epoch 00003: val_acc improved from 0.77270 to 0.82687, saving model to /tmp/model-task2.3.hdf5
#   Epoch 00004: val_acc improved from 0.82687 to 0.83696, saving model to /tmp/model-task2.3.hdf5
#   Epoch 00005: val_acc improved from 0.83696 to 0.84227, saving model to /tmp/model-task2.3.hdf5
#   Epoch 00006: val_acc improved from 0.84227 to 0.84440, saving model to /tmp/model-task2.3.hdf5
#   Epoch 00007: val_acc improved from 0.84440 to 0.84599, saving model to /tmp/model-task2.3.hdf5
#   Epoch 00008: val_acc did not improve
#   Epoch 00009: val_acc improved from 0.84599 to 0.84758, saving model to /tmp/model-task2.3.hdf5
#   Epoch 00010: val_acc did not improve
#   Epoch 00011: val_acc did not improve
#   Epoch 00011: early stopping
#   Accuracy (dev) of CNN: 0.847584
#   Accuracy (test) of CNN: 0.841742
