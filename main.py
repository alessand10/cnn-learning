import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from image_importer import *


# x_train = x_train.astype("float64") / 255.0
# x_test = x_test.astype("float64") / 255.0

model = keras.Sequential([
    keras.Input(shape=(64,64,3)),
    layers.Conv2D(32, (2, 2), padding='valid', activation='relu'),
    layers.MaxPooling2D(pool_size=(3,3)),
    layers.Conv2D(64, (2, 2), padding='valid', activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(128, (2, 2), padding='valid', activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='softmax'),
    layers.Dense(4, activation='softmax')
])

print(model.summary())
no_tumor = import_images_from_folder("C:\\Users\\Alessandro Genovese\\Downloads\\brain-tumors-dataset\\Testing\\single-pituitary", resize=(64,64))

# First 10 are cars, final 10 are random
tumorless = import_images_from_folder("C:\\Users\\Alessandro Genovese\\Downloads\\brain-tumors-dataset\\Training\\notumor", resize=(64,64))
pituitary = import_images_from_folder("C:\\Users\\Alessandro Genovese\\Downloads\\brain-tumors-dataset\\Training\\pituitary", resize=(64,64))
meningioma = import_images_from_folder("C:\\Users\\Alessandro Genovese\\Downloads\\brain-tumors-dataset\\Training\\meningioma", resize=(64,64))
glioma = import_images_from_folder("C:\\Users\\Alessandro Genovese\\Downloads\\brain-tumors-dataset\\Training\\glioma", resize=(64,64))

labels = np.array([0] * len(tumorless) + [1] * len(pituitary) + [2] * len(meningioma) + [3] * len(glioma))

all_training = np.concatenate((tumorless, pituitary, meningioma, glioma))


model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=3e-4),
    metrics=["accuracy"]
)

model.fit(all_training, labels, epochs=20)

prediction = model.predict(no_tumor)[0]
print('Tumorless: {}% | Pituitary: {}% | Meningioma: {}% | Glioma: {}%'.format(prediction[0], prediction[1], prediction[2], prediction[3]))
