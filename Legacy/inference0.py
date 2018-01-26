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


import os
import os.path
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Import Saved Model
model = Sequential()

def evalulate(model, image):
    img = cv2.imread(image)
    img = test.astype('float32')


model = load_model("cnn_mnist")

test = cv2.imread('test7.png')
# test =
test = test.astype('float32')


model.evaluate(test)
