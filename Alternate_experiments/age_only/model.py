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



def get_model(model_name="ResNet50"):
    base_model = None

    if model_name == "ResNet50":
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling="avg")

    if model_name == "Custom":
        model = Sequential()
        model.add(Conv2D(64, (3,3), input_shape=(224, 224, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(BatchNormalization())

        model.add(Conv2D(64, (3,3), input_shape=(224, 224, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(400))
        model.add(Activation('relu'))

        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dense(1))
        base_model = model
   
    prediction = Dense(units=101, kernel_initializer="he_normal", use_bias=False, activation="softmax",
                       name="pred_age")(base_model.output)

    prediction_2 = Dense(units=1, kernel_initializer="he_normal", use_bias=False, 
                       name="pred_age")(base_model.output)

    model = Model(inputs=base_model.input, outputs=prediction_2)
    
    return model



def main():
    model = get_model("ResNet50")
    model.summary()


if __name__ == '__main__':
    main()
