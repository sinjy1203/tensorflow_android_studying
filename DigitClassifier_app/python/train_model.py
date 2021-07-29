##
import math
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from util import *

mnist = keras.datasets.mnist.load_data()

##
(x_train, y_train), (x_test, y_test) = mnist
x_train = (x_train / 255.)[..., np.newaxis]
x_test = (x_test / 255.)[..., np.newaxis]

##
show_sample_data(x_train, y_train)

##
model = keras.models.Sequential()
model.add(keras.layers.Input(shape=(28, 28, 1)))
for i in range(3):
    model.add(keras.layers.Conv2D(filters=100, kernel_size=3,
                                  activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation='softmax'))

##
model.summary()

##
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=3)

##
test_loss, test_acc = model.evaluate(x_test, y_test)

##
pred = model.predict(x_test)
pred_arg = np.argmax(pred, axis=-1)

##
show_sample_data(x_test, pred_arg)

##
model_dir = Path('model')
model_dir.mkdir(exist_ok=True)

##
model.save(model_dir)

##
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

##
f = open('mnist.tflite', 'wb')
f.write(tflite_model)
f.close()

##

