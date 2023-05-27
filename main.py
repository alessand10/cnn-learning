import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from image_importer import *


# x_train = x_train.astype("float32") / 255.0
# x_test = x_test.astype("float32") / 255.0

model = keras.Sequential([
    keras.Input(shape=(32,32,3)),
    layers.Conv2D(32, (3, 3), padding='valid', activation='relu'),
    layers.MaxPooling2D(pool_size=(3,3)),
    layers.Conv2D(64, (3, 3), padding='valid', activation='relu'),
    layers.Flatten(),
    layers.Dense(32, activation='softmax'),
    layers.Dense(2, activation='softmax')
])

print(model.summary())
web_car = import_images_from_web(['https://files.porsche.com/filestore/image/multimedia/none/homepage-teaser-icc-718/normal/db4be0ec-f8f4-11eb-80db-005056bbdc38;sP;twebp/porsche-normal.webp'], resize=(32,32))

# First 10 are cars, final 10 are random
car_training = import_images_from_folder("C:\\Users\\Alessandro Genovese\\Downloads\\cars-dataset", resize=(32,32))
car_count = len(car_training)
random_training = import_images_from_folder("C:\\Users\\Alessandro Genovese\\Downloads\\random_image_dataset", resize=(32,32))
random_count = len(random_training)
labels = np.array([1] * car_count + [0] * random_count)

all_training = np.concatenate((car_training, random_training))

# model = keras.Sequential([
#     keras.Input(shape=(100, 100, 3)),
#     layers.Conv2D(1, (20, 20), strides=(5,5)),
#     layers.MaxPool2D(pool_size=(5,5), strides=(5,5)),
#     layers.Flatten(),
#     layers.Dense(2, activation='sigmoid')
# ])

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=3e-4),
    metrics=["accuracy"]
)

model.fit(all_training, labels, epochs=20)

print(model.predict(web_car))


# model = keras.Sequential([
#     layers.Input(shape=(1)),
#     layers.Dense(units=1, activation='relu',kernel_initializer=keras.initializers.RandomUniform(minval=0.0, maxval=1.0))
# ])

# model.compile(
#     loss = keras.losses.MeanSquaredError(),
#     optimizer=keras.optimizers.Adam(learning_rate=0.1),
#     metrics=["accuracy"]
# )

#print(model.summary())
