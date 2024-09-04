import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

"""
This file just tells you how accurate your model is. Printing the loss/accuracy
of your model.
"""

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)

x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.load_model('handwritten.keras')

loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)