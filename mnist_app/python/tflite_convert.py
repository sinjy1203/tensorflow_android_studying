##
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

model_dir = Path(".") / "model" / "mnist_model.h5"
model_pb_dir = Path(".") / "model_pb"
tflite_model_dir = Path(".") / "tflite_model"

##
model = keras.models.load_model(model_dir, compile=False)
model.save(model_pb_dir, save_format="tf")

##
converter = tf.lite.TFLiteConverter.from_saved_model(str(model_pb_dir))
tflite_model = converter.convert()

with open(tflite_model_dir / "mnist_model.tflite", 'wb') as f:
    f.write(tflite_model)

##

