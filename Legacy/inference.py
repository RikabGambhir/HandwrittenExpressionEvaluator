from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
import cv2
import numpy as np

import segmentation

import os
import os.path
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Import Saved Model

classes = ("0","1","2","3","4","5","6","7","8","9","+","-","*","/")

def evalulate(model, image):
    segments = segmentation.segment(image)
    predicted = []
    for i in range(len(segments)):
        digit = predict(model, segments[i])
        predicted.append(digit)
    return ''.join(predicted)

def predict(model, img):
    img = img[None,:,:,None]
    prediction = model.predict(img)
    return classes[int(np.argmax(prediction))]
