# training module
import modules.plots as plots
import os
import datetime
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten
from keras.callbacks import TensorBoard

def train(X_train, y_train, X_valid, y_valid, n_classes, logdir='/tmp/logs/deep-net/'):
  if not os.path.exists(logdir):
    os.mkdir(logdir)

  print(F"Logging to: {logdir}")
  print(F"Start training categprical with classes: {n_classes}")
  print(F"X_train shape: {np.shape(X_train)}")
  print(F"X_valid shape: {np.shape(X_valid)}")
  
  model = Sequential()
  
  # input layer - 2d matrix input
  model.add(Conv2D(32, kernel_size(3,3), activation='relu', input_shape=(28,28,1)))
  model.add(Conv2D(64, kernel_size(3,3), activation='relu'))
  # pool to reduce complexity
  model.add(MaxPooling2D(pool_size=(2,2)))
  # randomize forward prop dropout to reduce over-fit 
  model.add(Dropout(.25))
  # Flatten - convert 2D matrix to 1D matrix
  model.add(Dense(128, activation='relu'))
  # radomize forward prop dropout to reduce over-fit
  model.add(Dropout(.50))
  # output for categorization
  model.add(Dense(n_classes, activation='softmax'))

  model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

  #print summary
  model.summary()

  log_dir = logdir + datetime.datetime.now().strftime("Y%m%d-%H%M%S")
  
  tensorboard = TensorBoard(log_dir, histogram_freq=1)

  # train
  history = model.fit(X_train, y_train,
    batch_size=128,
    epochs=20,
    verbose=1,
    validation_data=(X_valid, y_valid),
    callbacks=[tensorboard])

  # render plots
  plots.show_keys(history)
  plots.plot_accuracy(history)
  plots.plot_loss(history)

  return model

