##
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
from util import *

zero_img_path = keras.utils.get_file(
    'zero.png',
    'https://storage.googleapis.com/khanhlvg-public.appspot.com/digit-classifier/zero.png'
)

image = keras.preprocessing.image.load_img(
    zero_img_path,
    color_mode='grayscale',
    target_size=(28, 28),
    interpolation='bilinear'
)

##
input_image = (np.array(image, dtype=np.float32) / 255.)[np.newaxis, ..., np.newaxis]

##
show_image(image, label=0)

##
tflite_path = "mnist.tflite"
with open(tflite_path, 'rb') as f:
    tflite_model = f.read()

##
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

##
interpreter.set_tensor(interpreter.get_input_details()[0]["index"], input_image)
interpreter.invoke()
output = interpreter.tensor(interpreter.get_output_details()[0]['index'])()[0]

##
pred_digit = np.argmax(output)

##
show_image(image, pred_digit)
print("confidence: {}".format(output[pred_digit]))
