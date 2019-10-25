import numpy as np
import pandas as pd
import csv
import sys
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import keras

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import load_model

def normalize_data(x, normalize_column):
    x = x.astype(np.float64)
    tmp = x[:, normalize_column]
    tmp = preprocessing.scale(tmp, axis=0)
    x[:, normalize_column] = tmp

    return x

x_train = pd.read_csv(sys.argv[3])
y_train = pd.read_csv(sys.argv[4], header = None)
x_train = x_train.values
y_train = y_train.values
y_train = y_train.reshape(-1)

x_test = pd.read_csv(sys.argv[5])
x_test = x_test.values

normalize_column = [0,1,3,4,5]
x_train = normalize_data(x_train, normalize_column)
x_test = normalize_data(x_test, normalize_column)

X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, test_size = 0.25, random_state = 0)

Y_train = np_utils.to_categorical(Y_train, 2)
Y_valid = np_utils.to_categorical(Y_valid, 2)

#Creating a model
model = Sequential()
model.add(Dense(input_dim=106, units=689, activation='relu'))
model.add(Dense(units=2, activation='softmax'))

# Compiling the model 
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=SGD(lr=0.1))

# Actual modelling
model.fit(X_train, Y_train, verbose=0, batch_size=100, epochs=20)

score, accuracy = model.evaluate(X_valid, Y_valid, batch_size=10000, verbose=0)
print(accuracy)

# creates a HDF5 file 'keras-model.h5'
model.save('keras-model.h5') 