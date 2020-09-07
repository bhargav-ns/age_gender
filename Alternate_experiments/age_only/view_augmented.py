import better_exceptions
import random
import math
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from keras.utils import Sequence, to_categorical
import Augmentor
import dlib
import argparse
from contextlib import contextmanager
import pdb
import os

"""
Pure Keras data generator function:

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

"""

# Data augmentation
def get_transform_func():
    p = Augmentor.Pipeline()
    p.flip_left_right(probability=0.5)
    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    p.zoom_random(probability=0.5, percentage_area=0.95)
    p.random_distortion(probability=0.5, grid_width=2, grid_height=2, magnitude=8)
    p.random_color(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_contrast(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_brightness(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_erasing(probability=0.5, rectangle_area=0.2)



    def transform_image(image):
        image = [Image.fromarray(image)]
        for operation in p.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                image = operation.perform_operation(image)
        return image[0]
    return transform_image

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

dir_size = 800
path = r"./RDM_Images_IMDB"
im_list = []
for oi in range(dir_size):
    random_filename = random.choice([
        x for x in os.listdir(path)
        if os.path.isfile(os.path.join(path, x))
    ])

    print(random_filename)
    new_path = path + "/" + random_filename

    detector = dlib.get_frontal_face_detector()
    transform_image = get_transform_func()
    
    img = cv2.imread(new_path)
    cv2.imshow("Original",img)
    
    

    l = np.array(transform_image(img))

    input_img = cv2.cvtColor(l, cv2.COLOR_BGR2RGB)
    img_size = 64
    margin = 0.4    
    label = 'detector_sample'

    img_h, img_w, _ = np.shape(input_img)

    # detect faces using dlib detector
    detected = detector(input_img, 1)
    faces = np.empty((len(detected), img_size, img_size, 3))


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

            cv2.rectangle(input_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            #cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
            cv2.rectangle(input_img, (xk1, yk1), (xk2, yk2), (255, 0, 0), 2)
            faces[i, :, :, :] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
            print(xw1, yw1,xw2,yw2)
    else:
        print("Not detected")
        

    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    cv2.imshow("Detected",input_img)
    cv2.waitKey(0)
    pdb.set_trace()