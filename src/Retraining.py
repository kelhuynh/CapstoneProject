"""
This module is used for the training of motion gesture recognition using a sequential
model consisting of LSTM and Dense layers.

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

import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import TensorBoard
from keras import regularizers
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


DATA_PATH = os.path.join('MP_Data')  # Define file path to dataset
actions = np.loadtxt("actions.txt", dtype='str')  # Load gesture labels into an array

# Thirty videos worth of data
no_sequences = 30

# Number of frames per video
sequence_length = 30

# Map dataset to gesture labels
# TODO: optimize label mapping, this part is most of the execution time
label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# Format dataset for model training
X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Build machine learning model
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, kernel_regularizer=regularizers.l2(0.01), return_sequences=True, activation='relu', input_shape=(30, 258)))
model.add(Dropout(0.2))
model.add(LSTM(128, kernel_regularizer=regularizers.l2(0.01), return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(64, kernel_regularizer=regularizers.l2(0.01), return_sequences=False, activation='relu'))
model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
model.add(Dense(32, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
#cp_callback = tf.keras.callbacks.ModelCheckpoint('action.h5', verbose=1, save_weights_only=False)  # noqa: E265
#es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)  # noqa: E265
model.compile(optimizer='Nadam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback, cp_callback, es_callback])  # noqa: E265
model.fit(X_train, y_train, epochs=100, callbacks=[tb_callback])

# Create confusion matrix for analysis of model accuracy
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
#multilabel_confusion_matrix(ytrue, yhat)  # noqa: E265
print(multilabel_confusion_matrix(ytrue, yhat))

model.save('action.h5')
