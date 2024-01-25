import os
import shutil

import tensorflow as tf
import cv2

num_classes = 2

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])

checkpoint_path = "./training_1/cp.ckpt"
model.load_weights(checkpoint_path)

directory = "./imgs"
files = os.listdir(directory)

if not(os.path.isdir('./imgs/cat')): os.mkdir('./imgs/cat')
if not(os.path.isdir('./imgs/dog')): os.mkdir('./imgs/dog')

for file in files:
    path = './imgs/{file}'.format(file=file)

    img = cv2.imread(path)
    img = cv2.resize(img, (180, 180))
    img = img.reshape((1, 180, 180, 3))

    output = model.predict(img)

    if (output[0][0] > output[0][1]): shutil.move(path, './imgs/cat/{file}'.format(file=file))
    else: shutil.move(path, './imgs/dog/{file}'.format(file=file))