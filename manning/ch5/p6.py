#!/usr/bin/env python

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

x = Input(shape=(28,28,1))
x1 = MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
x1 = Conv2D(64, (1,1), strides=(1,1), padding='same', activation='relu')(x1)
x2 = Conv2D(64, (1,1), strides=(1,1), padding='same', activation='relu')(x)
x3 = Conv2D(64, (1,1), strides=(1,1), padding='same', activation='relu')(x)
x3 = Conv2D(96, (3,3), strides=(1,1), padding='same', activation='relu')(x3)
x4 = Conv2D(64, (1,1), strides=(1,1), padding='same', activation='relu')(x)
x4 = Conv2D(48, (5,5), strides=(1,1), padding='same', activation='relu')(x4)

outputs = Concatenate()([x1,x2,x3,x4])
model = Model(x,outputs)

model.summary()
