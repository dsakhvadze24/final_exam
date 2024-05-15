import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from PIL import Image

# load the CIFAR-10 dataset
(train_images, train_labels), (_, _) = cifar10.load_data()

# Load gun and car images
gun_images = []
car_images = []

gun_path = "dataset/gun/"
car_path = "dataset/car/"

for filename in os.listdir(gun_path):
    img = load_img(gun_path + filename, target_size=(32, 32))
    img_array = img_to_array(img)
    gun_images.append(img_array)

for filename in os.listdir(car_path):
    img = load_img(car_path + filename, target_size=(32, 32))
    img_array = img_to_array(img)
    car_images.append(img_array)

gun_images = np.array(gun_images)
car_images = np.array(car_images)

# Assign labels
gun_labels = np.ones(len(gun_images))
car_labels = np.zeros(len(car_images))


# Combine images and labels
train_images = np.concatenate((train_images, car_images, gun_images), axis=0)
train_labels = np.concatenate((train_labels, car_labels, gun_labels), axis=0)


# Normalize pixel values to be between 0 and 1
train_images = train_images.astype('float32') / 255.0


# define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')  # Two classes: car and gun
])

# compile modeli
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train model
model.fit(train_images, train_labels, epochs=10, batch_size=64)
