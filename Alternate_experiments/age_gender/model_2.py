import better_exceptions
from keras.applications import ResNet50, InceptionResNetV2, MobileNet, Xception
from keras.layers import Dense
from keras.models import Model
from keras import backend as K

from random import randrange

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model, Sequential
from keras.layers import Dense
from keras.layers import Flatten
import tensorflow as tf

from tensorflow.keras.layers import Activation, Dropout, Conv2D, MaxPooling2D, BatchNormalization

# Mean average error for age
def mae2(y_true, y_pred):
    # true_age = K.sum(y_true * K.arange(0, 101, dtype="float32"), axis=-1)
    # pred_age = K.sum(y_pred * K.arange(0, 101, dtype="float32"), axis=-1)
    # mae = K.mean(K.abs(true_age - pred_age))
    return K.mean(K.abs(y_pred - y_true)) / K.mean(K.abs(y_true)) * 100/12
    #mae = tf.keras.losses.MeanAbsoluteError()
    #mae = mae/3

    #return mae

def mse(y_true, y_pred):
    # true_age = K.sum(y_true * K.arange(0, 101, dtype="float32"), axis=-1)
    # pred_age = K.sum(y_pred * K.arange(0, 101, dtype="float32"), axis=-1)
    # mae = K.mean(K.abs(true_age - pred_age))
    s = K.mean(K.abs(y_pred - y_true)) / K.mean(K.abs(y_true)) * 100/12
    return ((s^2*2))
    #mae = tf.keras.losses.MeanAbsoluteError()
    #mae = mae/3

    #return mae
