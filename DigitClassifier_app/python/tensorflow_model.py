##
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

from util import *

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

##
x_train = (x_train / 255.)[..., np.newaxis]
x_test = (x_test / 255.)[..., np.newaxis]

##
model_dir = Path('model')
model = keras.models.load_model(model_dir)

##
test_loss, test_acc = model.evaluate(x_test, y_test)

##
pred = model.predict(x_test)
pred_arg = np.argmax(pred, axis=-1)

show_sample_data(x_test, pred_arg)

##

