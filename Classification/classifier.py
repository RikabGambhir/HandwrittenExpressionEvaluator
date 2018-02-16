import sys, os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model


CLASSES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'plus', 'minus', 'multiplication', 'division1']
model = load_model('classification')

test_folder = 'test'
folder_name = 'division1'
image_number = '_0'


def predict_single_image(img):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    prediction_single = model.predict(img[None,:,:,None])
    prediction_single = prediction_single.argmax()
    prediction_single = CLASSES[prediction_single]

    fig = plt.figure(figsize = [10,10])
    fig.add_subplot(111)
    plt.imshow(img.reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted %s" % (prediction_single))
    plt.show()
    return prediction_single

def predict_multiple_images(test_folder):
    prediction_multiple_list = list()

    for filename in os.listdir(test_folder):
        images = cv2.imread(os.path.join(test_folder,filename))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
        prediction_multiple = model.predict(images[None,:,:,None])
        prediction_multiple = prediction_multiple.argmax()
        prediction_multiple = CLASSES[prediction_multiple]
        print("Predicted %s" % (prediction_multiple))
        prediction_string_name = str(prediction_multiple)
        print(prediction_string_name)
        prediction_multiple_list = prediction_multiple_list.append(prediction_string_name)
    print(prediction_multiple_list)

    for i in (len(prediction_multiple_list)):
        fig = plt.figure(figsize = [5,5])
        fig.add_subplot(3,3,i+1)
        plt.imshow(images.reshape(28,28), cmap='gray', interpolation='none')
        plt.title("Predicted %s" % (prediction_multiple_list[i]))
        plt.show()

    return prediction_multiple

#script_dir = sys.path[0]
#img_path = os.path.join(script_dir, folder_name + '/'+ image_number + '.png')
#prediction = predict_single_image(img_path)
#print("Predicted %s, Class %s" % (prediction,folder_name))


predictions = predict_multiple_images(test_folder)
