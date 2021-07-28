##
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist.load_data()

##
(x_train, y_train), (x_test, y_test) = mnist
x_train = (x_train / 255.)[..., np.newaxis]
x_test = (x_test / 255.)[..., np.newaxis]

##
def show_sample_data(x_train, y_train, sample_n=25):
    sample_x = np.squeeze((x_train[:25] * 255), -1)
    sample_y = y_train[:25]

    fig = plt.figure(figsize=(8, 7))
    rows = cols = math.sqrt(sample_n)

    idx = 1
    for x, y in zip(sample_x, sample_y):
        ax = fig.add_subplot(rows, cols, idx)
        ax.imshow(x, cmap='gray')
        ax.set_xlabel("target: {}".format(str(y)))
        ax.set_xticks([]), ax.set_yticks([])
        idx += 1

    plt.show()

##
show_sample_data(x_train, y_train)

##
model = keras.models.Sequential()
model.add(keras.layers.Input(shape=(28, 28, 1)))
for i in range(3):
    model.add(keras.layers.Conv2D(filters=100, kernel_size=3,
                                  activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10))

##
model.summary()
