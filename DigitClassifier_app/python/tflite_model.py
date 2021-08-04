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
show_image(image, label=0)

##
tflite_path = "mnist.tflite"
with open(tflite_path, 'rb') as f:
    tflite_model = f.read()

##

