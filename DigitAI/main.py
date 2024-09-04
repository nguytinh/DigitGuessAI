import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
"""
This file runs the model that we trained with the images I hand drew as a test.
The model is shit rn because I only trained it with 10 iterations of the dataset
"""

model = tf.keras.models.load_model('handwritten.keras')

image_number = 1
while os.path.isfile(f"numbers/num{image_number}.jpg"):
    try:
        img = cv2.imread(f"numbers/num{image_number}.jpg")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"The digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1