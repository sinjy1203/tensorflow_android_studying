##
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

data_dir = Path("mnist_data")
lst = list(data_dir.glob("*"))

##
image_pil = Image.open(lst[0]).convert("L")
image_arr = np.array(image_pil)
##
plt.imshow(image_arr)
plt.show()

##
class Mnist:
    def __init__(self, data_dir, model_path):
        self.data_dir = Path(data_dir)
        self.model_path = model_path

    def return_data(self):
        path_lst = list(self.data_dir.glob("*"))
        data_length = len(path_lst)
        image = np.zeros((data_length, 28, 28))
        label = np.zeros((data_length))

        for i in range(data_length):
            image_path = path_lst[i]
            image_arr = np.array(Image.open(image_path).convert("L"))
            image[i] = image_arr
            label[i] = int(image_path.name.split("_")[0])
        image = image[..., np.newaxis] / 255.
        return image, label

    def load_model(self):
        model = keras.models.load_model(self.model_path)
        return model

    def predict(self):
        model = self.load_model()
        image, label = self.return_data()
        return model.predict(image), label

    def accuracy(self):
        pred, label = self.predict()
        acc = keras.metrics.SparseCategoricalAccuracy()
        acc.update_state(label, pred)
        return acc.result()

##
m = Mnist("mnist_data", "model/mnist_model.h5")
accuracy = m.accuracy()

##
pred, label = m.predict()

##

