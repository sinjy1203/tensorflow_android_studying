##
import tensorflow as tf
from pathlib import Path

dir = Path.cwd().parent
model_dir = dir / "model" / "mnist_model.h5"
tflite_model_dir = dir / "tflite_model"

##
converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
tflite_model = converter.convert()

with open(str(tflite_model_dir / "mnist_model.tflite"), 'wb') as f:
    f.write(tflite_model)

##

