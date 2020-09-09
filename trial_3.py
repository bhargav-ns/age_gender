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
    return K.mean(K.abs(y_pred - y_true)) / K.mean(K.abs(y_true)) * 100
    #mae = tf.keras.losses.MeanAbsoluteError()
    #mae = mae/3

def convert_to_cat7(inp_labels):
    labels = list(map(int, inp_labels))
    new_labels = []
    label_list = ['Kids', 'Adol', 'Teen', 'Young Adult', 'Middle-aged Adult', '45-55', 'Old']
    for label in labels:
        if (label<(5)):
            new_label = 'Kids'
            new_labels.append(new_label)
        
        elif (label >= (5) and label < (12)):
            new_label = 'Adol'
            new_labels.append(new_label)

        elif (label >=(12) and label < (20)):
             new_label = 'Teen'
             new_labels.append(new_label)

        elif (label >=(20) and label < (35)):
             new_label = 'Young Adult'
             new_labels.append(new_label)

        elif (label >=(35) and label < (45)):
             new_label = 'Middle-aged Adult'
             new_labels.append(new_label)
        
        elif (label >=(45) and label < (55)):
             new_label = '45-55'
             new_labels.append(new_label)

        elif (label >=(55)):
             new_label = 'Old'
             new_labels.append(new_label)
        
        else:
            new_label = random.choice(label_list)
            new_labels.append(new_label)
        
    return new_labels

def convert_to_cat5(inp_labels):
    labels = list(map(int, inp_labels))
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
    p2 = h5py.File(layer_name + "/appa-real-release.hdf5", 'r')
    p3 = h5py.File(layer_name + "/All-Age-Faces-Dataset.hdf5", 'r')
    p4 = h5py.File(layer_name + "/part3.hdf5", 'r')
    p5 = h5py.File(layer_name + "/part1.hdf5", 'r')

    data_list = [p1, p2, p3, p4, p5]

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

def duplicate_teen_data(loaded_data):

    feature_set = loaded_data[0]
    label_set = np.ndarray.tolist(loaded_data[1])
    iter_set = label_set
    count = 0

    for i,label in tqdm(enumerate(iter_set), desc = "Augmenting set"):
        if (label >= 12 and label < 20):
            feature_set = np.concatenate((feature_set, feature_set[i].reshape(1,feature_set[0].shape[0],feature_set[0].shape[1],feature_set[0].shape[2])))
            label_set.append(label)
            print(i)
            count += 1
    print(count)
    return (feature_set, np.array(label_set))
pdb.set_trace()

# Extracting the labels and data

layer_name = "add_98"

random.seed(30)

loaded_data = load_layer_data(layer_name)
pdb.set_trace()

feature_set, label_set = duplicate_teen_data(loaded_data)


X = feature_set[0:int(np.round(feature_set.shape[0]*0.8))]
Y = label_set[0:int(np.round(feature_set.shape[0]*0.8))]
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

# Squashing all labels above 100 down to 100
Y = squash_labels(Y)
Y2 = Y
# Plotting a histogram for the distribution
m = np.ndarray.tolist(Y)
plt.hist(m, bins = 97)
plt.show()

pdb.set_trace()

Y = convert_to_cat5(Y)

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


model.add(Conv2D(128, (3,3)))
model.add(Activation("relu"))
model.add(BatchNormalization())
"""

model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))

# Use 7 units instead of 5 if you intend to use 7 categories
model.add(Dense(5, kernel_initializer="he_normal", use_bias=False, activation="softmax",
                       name="pred_age"))

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['acc', mae])



# Loading (optional) and training the model
"""
try:
    loaded_model = keras.models.load_model('models/' + layer_name)
    print('Model loaded')
    model = loaded_model
except:
    pass
"""

history = model.fit(X,dummy_y, epochs = 3, validation_split= 0.2)

model.save('models/' + layer_name)

pdb.set_trace()

model.metrics
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./results/' + layer_name + '/plot.png')
plt.show()

pdb.set_trace()

# Set cats to 1 if you want to use 7 categories instead of 5
cats = 0

def eval(cats):
    test_data = h5py.File(layer_name + "/appa-real-release.hdf5", 'r')
    test_labels = np.array(test_data.get('labels'))
    test_images = np.array(test_data.get('images'))

    test_images = feature_set[int(np.round(feature_set.shape[0]*0.8)): feature_set.shape[0]]
    test_labels = label_set[int(np.round(feature_set.shape[0]*0.8)): feature_set.shape[0]]

    test_labels = squash_labels(test_labels)

    if (cats == 0):
        test_labels = convert_to_cat5(test_labels)
    else:
        test_labels = convert_to_cat7(test_labels)

    encoder = LabelEncoder()
    encoder.fit(test_labels)
    encoded_Y2 = encoder.transform(test_labels)
    dummy_y2 = np_utils.to_categorical(encoded_Y2)

    y_pred = model.predict(test_images)

    matrix = confusion_matrix(dummy_y2.argmax(axis=1), y_pred.argmax(axis=1))
    final_mat = []

    sum_arr = []
    sum1 = 0

    for i in matrix.T:
        sum1 = 0
        for ele in i:
            sum1 += ele
        sum_arr.append(sum1)

    for j, row in enumerate(matrix.T):
        temp_mat = []
        for ele in row:
            temp_mat.append(float(ele/sum_arr[j]))
        final_mat.append(temp_mat)
            
    final_mat_list = np.around(np.array(final_mat).T, decimals = 3)

    for ele in final_mat:
        print(ele)

    print("Confusion Matrix:")
    print(matrix)

    if (cats == 1):
        pd.DataFrame(final_mat_list).to_csv("./results/" + layer_name + "/prob_conf_mat7.csv")
        pd.DataFrame(matrix).to_csv("./results/" + layer_name + "/raw_classification7.csv")
    else:
        pd.DataFrame(final_mat_list).to_csv("./results/" + layer_name + "/prob_conf_mat5.csv")
        pd.DataFrame(matrix).to_csv("./results/" + layer_name + "/raw_classification5.csv")

eval(cats)

pdb.set_trace()
