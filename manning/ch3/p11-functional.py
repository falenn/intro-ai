#!/usr/bin/env python

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

# create the input vector (128 x 128)
inputs = Input(shape=(128,128,1))
layer = Conv2D(16, kernel_size=(3,3), strides=(2,2,), padding="same", activation='relu')(inputs)
layer = MaxPooling2D(pool_size=(2,2), strides=(2,2))(layer)
layer = Flatten()(layer)
layer = Dense(512, activation='relu')(layer)
outputs = Dense(26, activation='softmax')(layer)

# Now, create the neural net
model = Model(inputs, outputs)

model.summary()
