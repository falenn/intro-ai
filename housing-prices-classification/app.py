#!/usr/bin/env python
import numpy as np
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization

# Boston housing prices in 1970
# About the data
#1.  Per capita crime rate.
#2.  Proportion of residential land zoned for lots over 25,000 square feet.
#3.  Proportion of non-retail business acres per town.
#4.  Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
#5.  Nitric oxides concentration (parts per 10 million).
#6.  Average number of rooms per dwelling.
#7.  Proportion of owner-occupied units built prior to 1940.
#8.  Weighted distances to five Boston employment centres.
#9.  Index of accessibility to radial highways.
#10. Full-value property-tax rate per $10,000.
#11. Pupil-teacher ratio by town.
#12. 1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
#13. % lower status of the population.


# Load Data
# y value is median house price in thousands of $$
# X is set of 13 features ranging from crime rate, to suburb for the building
(X_train, y_train), (X_valid, y_valid) = boston_housing.load_data()
print(F"Sample Data training dim: {X_train.ndim} shape: {X_train.shape}, size: {X_train.size}")
print(F"Sample Data training label dim: {y_train.ndim}, shape: {y_train.shape}, size: {y_train.size}")

print(F"Sample training data")
for k in range(3):
  print(F"{y_train[k]}: {X_train[k]}")

# model
model = Sequential()
# the input data is small, so we decide not to get over-zealous with the neurons
model.add(Dense(32, input_dim=13, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
# linear output to help predict a price (a continuous variable) - a non- bucketed or sigmoid value 
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

# Train the model
model.fit(X_train, y_train,
	batch_size=8,
	epochs=32,
	verbose=1,
	validation_data=(X_valid, y_valid))

# predict - grab record and reshape to input dim [1, 13]
out = model.predict(np.reshape(X_valid[42], [1, 13]))
print(F"predict value for {X_valid[42]}")
print()
print(F"value: {out}")

dim11 = [0]* X_valid.size
for k in range(X_valid.size):
  dim11[k] = X_valid[k][11]

print(F"Max num: {np.argmax(dim11)}")



vals = [9.329,0.,18.1,0.,0.713,6.185,98.7,2.2,24.,666.,20.2, 1000.0, 18.13 ]
myhouse = np.zeros((1,13))
myhouse[0] = vals
out = model.predict(myhouse)
print(F"myhouse: {out}")

