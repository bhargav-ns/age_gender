# Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model, Sequential
from keras.layers import Dense
from keras.layers import Flatten

from tensorflow.keras.layers import Activation, Dropout, Conv2D, MaxPooling2D, BatchNormalization

import better_exceptions
import random
import math
from pathlib import Path
from PIL import Image
import dlib

import pandas as pd
from keras.utils import Sequence, to_categorical
import Augmentor
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle
from pickle import dump
import cv2
import os
import random
from tqdm import tqdm
# -----
count = 0

# Loading the training data

DATADIR = "C:/Users/nsbha/Desktop/Projects/All_age_data/All-Age-Faces Dataset/All-Age-Faces Dataset/original images"
CATEGORIES = list(range(1,81))
labels = []

IMG_SIZE = 224

X = []
Y = []
K = []
training_data = []




# Shape of input - ((224,224,3),label)
def create_training_data(size):
    count = 0
    for img in tqdm(os.listdir(DATADIR), desc = "loading.."):
        try:
            img_array = cv2.imread(os.path.join(DATADIR, img))
            new_array = cv2.resize(img_array, (size, size))
            count += 1
            label = img[-6:-4]
            X.append(new_array)
            Y.append(label)
            K.append(int(label))
            training_data.append([new_array, float(label)])
        except Exception as e:
            print ("Error in loading " + img)


def get_transform_func():
    p = Augmentor.Pipeline()
    p.flip_left_right(probability=0.5)
    p.rotate(probability=0.2, max_left_rotation=5, max_right_rotation=5)
    p.zoom_random(probability=0.2, percentage_area=0.95)
    p.random_distortion(probability=0.2, grid_width=2, grid_height=2, magnitude=8)
    p.random_color(probability=0.2, min_factor=0.3, max_factor=0.5)
    p.random_contrast(probability=0.2, min_factor=0.8, max_factor=1.2)
    p.random_brightness(probability=0.2, min_factor=0.8, max_factor=1.2)
    p.random_erasing(probability=0.2, rectangle_area=0.2)

    def transform_image(image):
        image = [Image.fromarray(image)]
        for operation in p.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                image = operation.perform_operation(image)
        return image[0]
    return transform_image


create_training_data(IMG_SIZE)
detector = dlib.get_frontal_face_detector()

faces = []
training_data_aug = []


for im in tqdm(X, desc = "Loading.."):

    try:
        transform_image = get_transform_func()
        # input_img = np.array(transform_image(im))
        input_img = im
        img_size = 224
        margin = 0.4    
        label = 'detector_sample'

        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces_arr = np.empty((len(detected), img_size, img_size, 3))


        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)

                xk1 = max(int(x1 - 0.075 * w), 0)
                yk1 = max(int(y1 - 0.125 * h), 0)
                xk2 = min(int(x2 + 0.075 * w), img_w - 1)
                yk2 = min(int(y2 + 0.025 * h), img_h - 1)

                #cv2.rectangle(input_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                #cv2.rectangle(input_img, (xk1, yk1), (xk2, yk2), (255, 0, 0), 2)
                faces_arr[i, :, :, :] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                faces.append(faces_arr)
                training_data_aug.append([faces_arr, Y[i]])
                print(xw1, yw1,xw2,yw2)
        else:
            print("Not detected")
    
    except:
        pass
        

# Plotting the distribution of ages
plt.hist(K, bins=79)
plt.gca().set(title='Frequency Histogram for Ages', ylabel='Frequency')
plt.show()

training_dataK = np.array(X)
unshuffled_training_data_aug = training_data_aug

# Distribution list only contains the labels
dist_list = []

for num in CATEGORIES:
    counter = 0
    for label in K:
        if (label == num):
            counter += 1
    # To check how many instances of each label is present    
    dist_list.append([counter, num])


random.shuffle(training_data_aug)
X = []
Y = []

X1 = []
Y1 = []

print("List shuffled")
print("Recomputing X dimensions...")

pdb.set_trace()

# Making separate lists of features and labels
for features, label in training_data_aug:
    X.append(features)
    Y.append(label)

for features, label in unshuffled_training_data_aug:
    X1.append(features)
    Y1.append(label)


X = np.array(X).reshape(-1,IMG_SIZE, IMG_SIZE, 3)
X1 = np.array(X1).reshape(-1,IMG_SIZE, IMG_SIZE, 3)

pdb.set_trace()

print("Recomputed X")

# Pickling the prepared data for export and training
pickle_out = open('X2.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

# This is the unshuffled data
pickle_out = open('X1_unshuffled.pickle', 'wb')
pickle.dump(X1, pickle_out)
pickle_out.close()

pickle_out = open('Y.pickle', 'wb')
pickle.dump(Y, pickle_out)
pickle_out.close()

# Unshuffled labels
pickle_out = open('Y1_unshuffled.pickle', 'wb')
pickle.dump(Y1, pickle_out)
pickle_out.close()

X = pickle.load(open('X2.pickle', 'rb'))

# Data loading, storing, and sorting complete



