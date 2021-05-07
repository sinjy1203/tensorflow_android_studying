##
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path

##
data_dir = Path(".") / "mnist_data"

##
def make_image_path(label, order, data_dir):
    filename = "{}_{:03d}.png".format(label, order)
    return data_dir / filename

##
mnist = keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist

##
index_dic = dict()
for i in range(10):
    index = np.where(y_test == i)[0]
    index_dic[i] = index

##
for label in index_dic:
    for count, index in enumerate(index_dic[label][:5]):
        image = x_test[index]
        image_path = make_image_path(label, count, data_dir)
        plt.imsave(image_path, image, cmap='gray')
