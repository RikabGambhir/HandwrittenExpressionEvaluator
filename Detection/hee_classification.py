#from __future__ import print_function
import keras
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
# import clock

# start_time = clock.now()

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
keras.backend.set_session(sess)


CLASSES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'plus', 'minus', 'multiplication', 'division1']
NUM_CLASSES = len(CLASSES)  #The number of classes
IMG_SIZE = 28
BATCH_SIZE = 128	        # The number of images to process during a single pass
EPOCHS = 25	            # The number of times to iterate through the entire training set
IMG_ROWS, IMG_COLS = IMG_SIZE, IMG_SIZE                 # Input Image Dimensions
DATA_UTILIZATION = 1        # Fraction of data which is utilized in training and testing
TEST_RATIO = 1/6
dynamic_plotting = True

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename)) # Reads the images from folder
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # Converts images to black and white
        if img is not None:
            images.append(img)
    return images

# Define function to load data set and return testing and training data
def loadData(test_ratio = TEST_RATIO):
    data = np.zeros((0,IMG_ROWS, IMG_COLS))
    labels = []

    ###Load data from folder titled after the character class name eg. ('Zero', 'One', 'Two'...) and label it with a
    ###corresponding integer value eg. (0, 1, 2...)
    for i, CLASS in enumerate(CLASSES):
        images = load_images_from_folder(CLASS)       # Load images from folder
        print(CLASS +" loaded")



        data = np.concatenate((data,images), axis=0)    # Add features (images) to data variable
        # print("concatenated")

        label = len(images) * [i]                           # Create a list of the feature labels the length of the number of images
        labels = np.concatenate((labels, label), axis=0)    # Append the list of labels to the labels variable

    sort_data = list(zip(data,labels))                  # Zip the together the labels and features
    random.shuffle(sort_data)                           # Shuffle the labels and features together
    data, labels = zip(*sort_data)                      # Unzip the labels-features variable back into data and labels

    # Delete proportion of data equal to 1-DATA_UTILIZATION to speed up training and testing
    data = data[0:int(len(data)*DATA_UTILIZATION)]
    labels = labels[0:int(len(data)*DATA_UTILIZATION)]

    # Split data into test and train sets
    cutoff = int((1 - TEST_RATIO) * len(data))          # Determine the index at which to split the dataset into test and train
    x_train = data[0:cutoff]                            # Training features
    y_train = labels[0:cutoff]                          # Training labels
    x_test = data[cutoff:]                              # Testing features
    y_test = labels[cutoff:]                            # Testing labels

    return (np.asarray(x_train), np.asarray(y_train)), (np.asarray(x_test), np.asarray(y_test))

# Load data
(x_train, y_train), (x_test, y_test) = loadData(test_ratio = TEST_RATIO)
# (x_train, y_train), (x_test, y_test) = mnist.load_data()


x_test = x_test.reshape(x_test.shape[0],IMG_ROWS, IMG_COLS,1)     # Reshape x_test where 1 = number of colors
x_train = x_train.reshape(x_train.shape[0],IMG_ROWS, IMG_COLS,1)  # Reshape x_test
input_shape = (IMG_ROWS, IMG_COLS,1)
x_train = x_train.astype('float32')     # Convert x_train to float32
x_test = x_test.astype('float32')       # Convert x_test to float 32

x_train /= 255      # Scale feature values from 0-255 to values from 0-1
x_test /= 255       # Scale feature values from 0-255 to values from 0-1

print('x_train shape: ', x_train.shape)
print('y_train shape: ', y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to train = keras.utils.to_categorical(y_train,NUM_CLASSES = None) to binary class matrices
# Arguments: y: Class vector to be converted into a matrix (integers from 0 to num_classes)
#           num_classes: total number of classes
y_train = keras.utils.to_categorical(y_train,num_classes = None)
y_test = keras.utils.to_categorical(y_test, num_classes = None)

NODES = 64
LAYERS = 2
#Define network
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Dropout(0.50))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.50))

# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(Dropout(0.50))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(Dropout(0.50))


model.add(Flatten())
for x in list(range(LAYERS)):
    model.add(Dense(NODES, activation='relu'))
model.add(Dropout(0.5))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(NUM_CLASSES, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

scores = []
from matplotlib import pyplot as plt
n=100

# Options for Dynamic Plotting
if dynamic_plotting:
    plt.axis([0,n,0,2])
    plt.ion()
    plt.show()
    plt.plot([0, n],[1, 1], 'k', linestyle='--')

val_losses = []
val_accs = []
losses = []
accs = []
def plot_metrics(val_accs=val_accs,val_losses=val_losses,accs=accs,losses=losses):
    epochs = list(range(0, len(val_accs)))
    plt.plot(epochs, val_accs, 'b', label='Validation Accuracy')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.plot(epochs, accs, 'c', label='Training Accuracy')
    plt.plot(epochs, losses, 'm', label='Training Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Loss and Accuracy Plot\nepochs = {0}, nodes = {1}, layers = {2}'.format(n, NODES, LAYERS))
    if dynamic_plotting:
        plt.draw()
        plt.pause(0.001)
    else:
        plt.show()

history = model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=2,
          validation_data=(x_test, y_test))
val_accs.append(history.history['val_acc'])
print(len(val_accs))
val_losses.append(history.history['val_loss'])
accs.append(history.history['acc'])
losses.append(history.history['loss'])

'''
for x in range(0,n):
    history = model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=1,
              verbose=2,
              validation_data=(x_test, y_test))
    val_accs.append(history.history['val_acc'])
    print(len(val_accs))
    val_losses.append(history.history['val_loss'])
    accs.append(history.history['acc'])
    losses.append(history.history['loss'])
'''

    # Dynamic Plotting
    # plot_metrics()

try:
    model.save('classification')
except:
    print("Couldn't Save!")

# Save data to file log
final_val_accuracy = history.history['val_acc']
final_train_accuracy = history.history['acc']
final_val_loss = history.history['val_loss']
final_train_loss = history.history['loss']
# training_time = clock.now() - start_time
with open('Data_Log.txt','a') as Data_Log:
    Data_Log.write('{0},{1},{2},{3},{4},{5},{6},{7}'.format(n, NODES, LAYERS, final_val_accuracy, final_val_loss, final_train_accuracy, final_train_loss, training_time))

# Plot results at end when dynamic plotting is off
if not dynamic_plotting:
    plt.axis([0,n,0,4])
    plt.xlabel('Epoch')
    plt.plot(list(range(0, n)), val_accs,'b')
    plt.plot(list(range(0, n)), val_losses, 'r')
    plt.plot(list(range(0, n)), accs, 'g')
    plt.plot(list(range(0,n)), losses, 'p')
    plt.plot([0,len(scores)],[1,1], 'k')
    plt.show()
else:
    input('Press [enter] to Continue: ')
