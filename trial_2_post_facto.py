import tensorflow as tf
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
import keras.backend as K

from tensorflow.keras.layers import Activation, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from collections import defaultdict 
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle
from pickle import dump
import cv2
import os
import random
import tensorboard
import datetime
from tqdm import tqdm
import h5py
import math

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import time
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# Squashing all ages above 100 to 100
def squash_labels(Y):
    for i,label in enumerate(Y):
        if label > 100:
            Y[i] = 100
            print(i)
    return Y

def mae(y_true, y_pred):
    # true_age = K.sum(y_true * K.arange(0, 101, dtype="float32"), axis=-1)
    # pred_age = K.sum(y_pred * K.arange(0, 101, dtype="float32"), axis=-1)
    # mae = K.mean(K.abs(true_age - pred_age))
    return K.mean(K.abs(y_pred - y_true)) / K.mean(K.abs(y_true)) * 10
    #mae = tf.keras.losses.MeanAbsoluteError()
    #mae = mae/3

def create_labels(labels):
    labels = list(map(int, labels))
    new_labels = []
    label_list = ['Child', 'Teen', 'Young Adult', 'Middle-aged Adult', 'Old']
    for label in labels:
        if (label<12):
            new_label = 'Child'
            new_labels.append(new_label)
        
        elif (label >= 12 and label < 20):
            new_label = 'Teen'
            new_labels.append(new_label)

        elif (label >=20 and label < 35):
             new_label = 'Young Adult'
             new_labels.append(new_label)

        elif (label >=35 and label < 55):
             new_label = 'Middle-aged Adult'
             new_labels.append(new_label)

        elif (label >= 55):
             new_label = 'Old'
             new_labels.append(new_label)
        
        else:
            new_label = random.choice(label_list)
            new_labels.append(new_label)
    return new_labels

def load_layer_data(layer_name):
    p1 = h5py.File(layer_name + "/part2.hdf5", 'r')
    p2 = h5py.File(layer_name + "/part1.hdf5", 'r')
    p3 = h5py.File(layer_name + "/All-Age-Faces-Dataset.hdf5", 'r')
    p4 = h5py.File(layer_name + "/part3.hdf5", 'r')

    data_list = [p1, p2, p3, p4]

    for i,ele in tqdm(enumerate(data_list), desc = "loading_data"):
        lab = np.array(ele.get('labels'))
        im = np.array(ele.get('images'))
        if i == 0:
            labels = lab
            images = im
        else:
            labels = np.concatenate((labels, lab))
            images = np.concatenate((images, im))

    return (images, labels)

pdb.set_trace()

# Extracting the labels and data

layer_name = "stage3_unit15_conv1"

X, Y = load_layer_data(layer_name)
pdb.set_trace()
  
def_dict = defaultdict(lambda: "Not Present") 

for j in range(110):
    count = 0
    for label in Y:
        if label == j:
            count += 1
        def_dict[str(j)] = count        
    if count == 0:
        print(j)

Y = squash_labels(Y)

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)

pdb.set_trace()

model = Sequential()

"""
model.add(Conv2D(256, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size = (3,3)))
model.add(BatchNormalization())
"""

model.add(Conv2D(128, (3,3)))
model.add(Activation("relu"))
model.add(BatchNormalization())

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(400))
model.add(Activation('relu'))

model.add(Dense(97, kernel_initializer="he_normal", use_bias=False, activation="softmax",
                       name="pred_age"))

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['acc', mae])

# Loading (optional) and training the model
"""
try:
    loaded_model = keras.models.load_model('models_postfacto/' + layer_name)
    print('Model loaded')
    model = loaded_model
except:
    pass
"""

history = model.fit(X,dummy_y, epochs = 30, validation_split= 0.2)
model.save('models_postfacto/' + layer_name)

pdb.set_trace()

# Just some plotting stuff
model.metrics
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./results_postfacto/' + layer_name + '/plot.png')
plt.show()

pdb.set_trace()

# Use the functions below to evaluate and save the confusion matrix to file

def eval():
    test_data = h5py.File(layer_name + "/appa-real-release.hdf5", 'r')
    test_labels = np.array(test_data.get('labels'))
    test_images = np.array(test_data.get('images'))

    test_labels = squash_labels(test_labels)
    y_pred = model.predict(test_images)

    matrix = confusion_matrix(create_labels(test_labels), create_labels(y_pred.argmax(axis = 1)))
    
    return matrix

def save_conf_mats(): 
    k = eval()   
    final_mat = []

    sum_arr = []
    sum1 = 0

    for i in k.T:
        sum1 = 0
        for ele in i:
            sum1 += ele
        sum_arr.append(sum1)

    for j, row in enumerate(k.T):
        temp_mat = []
        for ele in row:
            temp_mat.append(float(ele/sum_arr[j]))
        final_mat.append(temp_mat)
  
    final_mat = np.around(np.array(final_mat).T, decimals=3)
    dat_frame = pd.DataFrame({'Child': k[:, 0], 'Teen': k[:, 1],'Young Adult':k[:,2], 'Middle-aged Adult':k[:,3], 'Old':k[:,4]}, 
                        index = ['Child', 'Teen', 'Young Adult', 'Middle-aged Adult', 'Old'])
    dat_frame2 = pd.DataFrame({'Child': final_mat[:, 0], 'Teen': final_mat[:, 1],'Young Adult':final_mat[:,2], 'Middle-aged Adult':final_mat[:,3], 'Old':final_mat[:,4]}, 
                        index = ['Child', 'Teen', 'Young Adult', 'Middle-aged Adult', 'Old'])

    dat_frame.to_csv("./results_postfacto/" + layer_name + "/raw_classification.csv")
    dat_frame2.to_csv("./results_postfacto/" + layer_name + "/prob_conf_mat.csv")

print(eval())
save_conf_mats()

pdb.set_trace()