#!/usr/bin/env python
import os.path
import keras
from keras.datasets import mnist  # handwriting set from NIST
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import model_from_yaml
from matplotlib import pyplot as plt
import numpy as np

large_width = 400
np.set_printoptions(linewidth=large_width)



# Load data  
#X_train shape: (60000, 28, 28) - 60,000 images of 28x28 pixels
#y is used for labels
(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

print(F"training img 0[{y_train[0]}]: {X_train[0]}")

# plot some of these - save to disk to see
plt.figure(figsize=(5,5))
for k in range(12):
  plt.subplot(3,4, k+1)
  plt.imshow(X_train[k], cmap='Greys')
  plt.axis('off')
plt.tight_layout()
plt.savefig("figs/figs.png")
plt.show()

# Render a validation image
print(F"validation img 0[{y_valid[0]}: {X_valid[0]}")

def train(train, valid):
  X_train = train[0]
  y_train = train[1]

  X_valid = valid[0]
  y_valid = valid[1]

  # having a shallow neural net, we flatten our 2d images each into an array of 784-element 1d arrays, and convert datatype to float for division step 
  X_train = X_train.reshape(60000, 784).astype('float32')
  X_valid = X_valid.reshape(10000, 784).astype('float32')

  # convert greyscale value per pixel to a ratio from 0 -> 1
  X_train /= 255
  X_valid /= 255

  # convert integer labels to one-hot
  n_classes = 10
  y_train = keras.utils.to_categorical(y_train, n_classes)
  y_valid = keras.utils.to_categorical(y_valid, n_classes)
  print(F"one-hot representation[0]: {y_valid[0]}")

  # Architecting the DNN
  # Network type - sequential. each layer passes info to only the next layer in the sequence of layers in this DNN.
  model = Sequential()
  # add a hidden layer - add sigmoid-type artificial neurons in general-purpose, fully-connected arrangement (a.k.a, Dense)
  # This includes specifying the input shape of 784, - 1d array.
  model.add(Dense(64, activation='sigmoid', input_shape=(784,)))

  # Add next layer - our output layer with 10 aritifcial neurons of type softmax (to map to labels) using cooresponding probabilities
  model.add(Dense(10, activation='softmax'))

  # Compile the model
  model.compile(loss='mean_squared_error', optimizer='adam')


  #---- Training ----
  # fit - method to train the model
  # epochs - number of times to train - the passthrough of the 60,000 images is one epoch
  # after each epoch, validation occurs.  we should see validation improve over time
  model.fit(X_train, y_train, 
	batch_size=128, 
	epochs=10, 
	verbose=1, 
	validation_data=(X_valid, y_valid))

  # Save model to disk
  # serialize model to YAML
  model_yaml = model.to_yaml()
  with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
  # serialize weights to HDF5
  model.save_weights("model.h5")
  print("Saved model to disk")

if not os.path.exists('model.yaml'):
  print("No model exists.  Training....")
  train((X_train, y_train), (X_valid, y_valid)) 

# load YAML and create model
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(
	loss='mean_squared_error', 
	optimizer='adam', 
	metrics=['accuracy'])

X_validrs = X_valid.reshape(10000, 784).astype('float32')
X_validrs /= 255
y_valid = keras.utils.to_categorical(y_valid, 10)

score = loaded_model.evaluate(X_validrs, y_valid, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# now predict one of them
X_0 = X_valid[0]
X_0 = X_0.reshape(1, 784).astype('float32')
X_0 /= 255

probs = loaded_model.predict(X_0)
np.set_printoptions(precision=2)  
print(F"What is this? {X_valid[0]}")
print()
print(F"{probs}")

plt.figure(figsize=(5,5))
plt.subplot(3,4, 1)
plt.imshow(X_valid[0], cmap='Greys')
plt.axis('off')
plt.show()
plt.savefig("figs/x_valid0.png")




