import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
"""
This is the file that trains the model with the dataset already provided to us by
tensorflow in the keras directory. There are a lot of weird functions to get used to.
You can copy/paste code into chat gpt to get a better understanding.
"""

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)

x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20)

model.save('handwritten.keras')