#!/usr/bin/env python
import os.path
import modules.plots as Plots
import keras
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
  model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28, 28,1)))
  model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
  # pool to reduce complexity - 
  model.add(MaxPooling2D(pool_size=(2,2)))
  # randomize path backprop tuning for training balance to prevent over-fit
  model.add(Dropout(0.25))
  # convey many n-dimensional into 1-dimensional aray
  model.add(Flatten())
  # dense hidden
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  # output for categorial invariant
  model.add(Dense(n_classes, activation='softmax'))

  model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

  history = model.fit(X_train, y_train,
    batch_size=128,
    epochs=20,
    verbose=1,
    validation_data=(X_valid,y_valid))

  # plot training 
  plots.show_keys(history)
  plots.plot_accuracy(histoy)
  plots.plot_loss(history)
  
  # save model
  with open('model.yaml', 'w') as yaml_file:
    yaml_file.write(model.to_yaml())
  model.save_weights('model.h5')

else:
  # load model
  yaml_file = open('model.yaml', 'r')
  model = model_from_yaml(yaml_file.read())
  yaml_file.close()
  
model.summary()

X_0 = X_valid[0]
X_0 = X_0.reshape(28, 28).astype('float32')
X_0 /= 255
probs = model.predict(X_0)
print(F"{probs}")
print(F"max prob: {np.maxarg(probs)}")
