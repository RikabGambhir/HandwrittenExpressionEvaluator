'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import cv2
import numpy as np
import random


import os
import os.path
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is not None:
            images.append(img)
    return images



# Load dataset
def loadData(test_ratio = 1/6):
    data = np.zeros((0,28,28))
    labels = []

    for i in range(14):
        images = (load_images_from_folder( str(i)))
        print("loaded")

        data = np.concatenate((data,images), axis = 0)
        print("concatenated")


        l = len(images) * [i]
        labels = np.concatenate((labels, l), axis = 0)

    c = list(zip(data,labels))
    random.shuffle(c)
    data, labels = zip(*c)

    cutoff =  int((1 - test_ratio) * len(data))
    X_train = data[0: int(cutoff/3)]
    Y_train = labels[0: int(cutoff/3)]
    X_test = data[cutoff:]
    Y_test = labels[cutoff:]
    return (np.asarray(X_train), np.asarray(Y_train)), (np.asarray(X_test), np.asarray(Y_test))


batch_size = 128	# The number of images to process during a single pass
num_classes = 14	# The number of possible labels ("0","1",..."9")
epochs = 50	   # The number of times to iterate through the entire training set

# Input Image Dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

plus = load_images_from_folder("10")
plus6 = plus[5000:6000]
plus = plus[0:5000]

x_train = np.concatenate((x_train, plus), axis = 0)
y_train = np.concatenate((y_train, len(plus)*[10]), axis = 0)
x_test = np.concatenate((x_test, plus6), axis = 0)
y_test = np.concatenate((y_test, len(plus6)*[10]), axis = 0)

plus = load_images_from_folder("11")
plus6 = plus[5000:6000]
plus = plus[0:5000]

x_train = np.concatenate((x_train, plus), axis = 0)
y_train = np.concatenate((y_train, len(plus)*[11]), axis = 0)
x_test = np.concatenate((x_test, plus6), axis = 0)
y_test = np.concatenate((y_test, len(plus6)*[11]), axis = 0)

plus = load_images_from_folder("12")
plus6 = plus[5000:6000]
plus = plus[0:5000]

x_train = np.concatenate((x_train, plus), axis = 0)
y_train = np.concatenate((y_train, len(plus)*[12]), axis = 0)
y_test = np.concatenate((y_test, len(plus6)*[12]), axis = 0)
x_test = np.concatenate((x_test, plus6), axis = 0)


plus = load_images_from_folder("13")
plus6 = plus[5000:6000]
plus = plus[0:5000]

x_train = np.concatenate((x_train, plus), axis = 0)
y_train = np.concatenate((y_train, len(plus)*[13]), axis = 0)
x_test = np.concatenate((x_test, plus6), axis = 0)
y_test = np.concatenate((y_test, len(plus6)*[13]), axis = 0)


x_test = np.concatenate((x_test, plus6), axis = 0)
y_test = np.concatenate((y_test, len(plus6)*[10]), axis = 0)

# (x_train, y_train), (x_test, y_test) = loadData(test_ratio = 1/6)
print(x_train.shape)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(28, kernel_size=(3, 3),
                 activation='relu',
                 input_shape= input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

model.save('cnn_mnist')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
