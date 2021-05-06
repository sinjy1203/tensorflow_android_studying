##
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path

train_x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
train_y = np.array([0, 0, 0, 1])
train_y = train_y[:, np.newaxis]

model = keras.models.Sequential([
    keras.layers.Dense(1, activation='sigmoid', input_shape=[2])
])

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.5),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(train_x, train_y, epochs=100,
          callbacks=[keras.callbacks.EarlyStopping(monitor='loss',
                                                   patience=3)])
##
model_2 = keras.models.Sequential([
    model,
    keras.layers.Lambda(lambda x: tf.cast(x >= 0.5, dtype=tf.int32))
])

##
def tflite_convert(model, model_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(model_path, 'wb') as f:
        f.write(tflite_model)

dir = Path.cwd().parent
tflite_model_dir = dir / "tflite_model"

tflite_convert(model, tflite_model_dir / "And_score.tflite")
tflite_convert(model_2, tflite_model_dir / "And_logit.tflite")