#!/usr/bin/env python

# example for training on grayscale sign language images - softmax for 26 output nodes - letters of of the alphabet

# Keras's Neural Network components
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Kera's Convolutional Neural Network components
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten

model =Sequential()

# Create a convolutional layer with 16 3x3 filters and stride of two as the input 
# layer.
model.add(Conv2D(16,kernel_size=(3,3),strides=(2,2),padding="same",activation='relu',input_shape=(128,128,1)))
# Add a pooling layer to max pool (downsample) the feature maps into smaller pooled 
# feature maps.
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# Add a flattening layer to flatten the pooled feature maps to a 1D input vector # for the DNN.
model.add(Flatten())

# Create the input layer for the DNN, which is connected to the flattening layer of 
# the convolutional front-end.
model.add(Dense(512,activation='relu'))
model.add(Dense(26,activation='softmax'))
# Use the Categorical Cross Entropy loss function for a Multi-Class Classifier.

model.compile(loss='categorical_crossentropy',
	optimizer='adam',
	metrics=['accuracy'])

model.summary()
