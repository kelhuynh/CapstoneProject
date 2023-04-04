"""
This module is used for the training of static gesture recognition.

MIT License

Copyright (c) 2023 Kelvin Huynh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__authors__ = ["Fred Zhu", "Jiahui Chen", "Kelvin Huynh", "Mirza Nafi Hasan", "Robert Zhu", "Zifan Meng"]
__date__ = "2023/04/04"
__deprecated__ = False
__license__ = "MIT License"
__status__ = "Prototype"
__version__ = "1.0"

import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split


RANDOM_SEED = 51  # Set random seed to produce same output across multiple runs

# Define coordinate dataset and ML model locations
dataset = os.getcwd() + '\\src\\model\\keypoint_classifier\\keypoint.csv'
model_save_path = os.getcwd() + '\\src\\model\\keypoint_classifier\\keypoint_classifier.hdf5'

NUM_CLASSES = 27  # Change as more signs/classifications are added

# Load and format dataset for training
x_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))
x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)


# Build machine learning model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 2, )),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

# Define checkpoint and earlystopping callbacks for use
cp_callback = tf.keras.callbacks.ModelCheckpoint(model_save_path, verbose=1, save_weights_only=False)
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

# Configure model for training
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model using dataset
model.fit(
    x_train,
    y_train,
    epochs=400,
    batch_size=128,
    validation_data=(x_test, y_test),
    callbacks=[cp_callback]
)

# Model testing and validation
val_loss, val_acc = model.evaluate(x_test, y_test, batch_size=128)  # Retrieve loss and accuracy metrics from model
model = tf.keras.models.load_model(model_save_path)

predict_result = model.predict(np.array([x_test[0]]))
print(np.squeeze(predict_result))
print(np.argmax(np.squeeze(predict_result)))

model.save(model_save_path, include_optimizer=False)

# Convert to tflite model from Keras model
tflite_save_path = os.getcwd() + '\\src\\model\\keypoint_classifier\\keypoint_classifier.tflite'

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

open(tflite_save_path, 'wb').write(tflite_quantized_model)

interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], np.array([x_test[0]]))
interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]['index'])

print(np.squeeze(tflite_results))
print(np.argmax(np.squeeze(tflite_results)))
