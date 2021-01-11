#!/usr/bin/env python

import keras
from keras.datasets import mnist  # handwriting set from NIST
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot as plt
import numpy as np

large_width = 400
np.set_printoptions(linewidth=large_width)

# Load data
#X_train shape: (60000, 28, 28) - 60,000 images of 28x28 pixels
#y is used for labels
(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

print(F"training img 1[{y_train[1]}]: {X_train[1]}")

# plot some of these - save to disk to see
plt.figure(figsize=(5,5))
for k in range(12):
  if not os.path.exists("figs/fig_%s.png" % k):
    plt.subplot(3,4, k+1)
    plt.savefig("figs/fig_%s.png" % k)

