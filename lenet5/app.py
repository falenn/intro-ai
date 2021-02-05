#!/usr/bin/env python
import os.path
import modules.trainer as trainer
import modules.modelUtil as mu
import keras
import math
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten

# since the first layer of this network is convolutional, we can leave
# the import as a 2d matrix instead of stringing it out as a 1d array
# X_train shape (28,28) 60,000 rows
#Y_train as labels
(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

# convert ints to floats to scale between 0, 1
X_train = X_train.reshape(60000, 28, 28, 1).astype('float32')
X_valid = X_valid.reshape(10000, 28, 28, 1).astype('float32')

# scale
X_train /= 255
X_valid /= 255

# convert labeles to one-hot
n_classes = 10
y_train = keras.utils.to_categorical(y_train, n_classes)
y_valid = keras.utils.to_categorical(y_valid, n_classes)

model = Sequential()

# train if model doesn't exist
if not os.path.exists('model.yaml'):
  model = trainer(X_train, y_train, X_valid, y_valid, n_classes)
  mu.saveModel(model, 'model.yaml', 'model.h5')
else:
  # load model
  model = mu.loadModel('model.yaml','model.h5')
 
model.summary()

# do preditions
X_0 = X_valid[0]
X_0 = X_0.reshape(28, 28).astype('float32')
X_0 /= 255
probs = model.predict(X_0)
print(F"{probs}")
print(F"max prob: {np.maxarg(probs)}")
