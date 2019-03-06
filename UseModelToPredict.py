import cv2
import tensorflow as tf
import numpy as np

CATEGORIES = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

def prepare(filepath):
    IMG_SIZE = 32
    img_array = cv2.imread(filepath)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)/255.0


model = tf.keras.models.load_model("CNN.model")

prediction = model.predict([prepare('dog.jpg')])
print(CATEGORIES[np.argmax(prediction)])

prediction = model.predict([prepare('cat.jpg')])
print(CATEGORIES[np.argmax(prediction)])

prediction = model.predict([prepare('bird6.png')])
print(CATEGORIES[np.argmax(prediction)])
#print(CATEGORIES[int(prediction[0])])