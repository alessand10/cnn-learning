import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from image_importer import *
import matplotlib.pyplot as plt
import time


model = keras.Sequential([
    layers.InputLayer(input_shape=(1)),
    layers.Dense(units=1, activation='relu', kernel_initializer=keras.initializers.RandomUniform(minval=0.0, maxval=1.0))
])

print(model.summary())

model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=0.1))

x_train = np.array([1,2,3,4,5])
y_train = np.array([2, 4, 6, 8, 10])

model.fit(x_train, y_train, epochs=200)
print(model.get_weights())
print(model.predict(([15])))