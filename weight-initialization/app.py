#!/usr/bin/env python
# Standard imports
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, Activation
from keras.initializers import Zeros, RandomNormal
from keras.initializers import glorot_normal, glorot_uniform

# setup
# simulate 784 pixel values as inputs to a single dense layer of artificial nuerons
n_input = 784
# number of neurons
n_dense = 256
# initialize w & b (weights and biases) with reasonably scaled parameters
# Large w & b values tend to correspond to larger z values and therefore saturated 
# neurons.  Avoid these large values at the start (forms unmerited opinions of 
#relationship between x and y)
# Small or zero values perform the opposite and form unmerited aversion
# If weights and biases are both zero, this would wreck stochastic gradient descent.
b_init = Zeros()
w_init = RandomNormal(stddev=1.0)

# Sim
model = Sequential()
model.add(Dense(n_dense, 
		input_dim=n_input,
		kernel_initializer=w_init,
		bias_initializer=b_init))
model.add(Activation('sigmoid'))

model.summary()

# Generate first input
x = np.random.random((1,n_input))

# run frist data through
a = model.predict(x)

graph = plt.hist(np.transpose(a))
plt.savefig('init_hist.png', bbox_inches='tight')





