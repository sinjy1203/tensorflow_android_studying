##
import tensorflow as tf
from tensorflow import keras
import numpy as np

mnist = keras.datasets.mnist.load_data()

##
(x_train, y_train), (x_test, y_test) = mnist
x_train, x_valid = x_train[:50000], x_train[50000:]
y_train, y_valid = y_train[:50000], y_train[50000:]

##
x_train = x_train[..., np.newaxis] / 255.
x_valid = x_valid[..., np.newaxis] / 255.
x_test = x_test[..., np.newaxis] / 255.


##
model = keras.models.Sequential([
    keras.layers.Conv2D(100, 3, 2, 'same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(100, 3, 1, 'same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(100, 3, 2, 'same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(100, 3, 1, 'same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='Nadam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],)

model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_valid, y_valid),
          callbacks=[keras.callbacks.EarlyStopping(patience=2)])


##
from pathlib import Path
dir = Path.cwd().parent
model_dir = dir / "model"

##
model.save(model_dir / "mnist_model.h5")
