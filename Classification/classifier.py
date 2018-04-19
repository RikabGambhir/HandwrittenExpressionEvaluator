import sys, os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model
import math
from math import sqrt


CLASSES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'plus', 'minus', 'multiplication', 'division1']
model = load_model('classification')

# *** Input images
# test_folder = 'test'            # used for predict_multiple_images
# folder_name = 'division1'       # input folder name of single image
# image_number = '_0'

# *** Use this function to input images ==> outputs prediction
def predict_single_image(img):

    # img = cv2.imread(img, 0)                                # Reads the images in the folder and converts it to black and white

    prediction_single = model.predict(img[None,:,:,None])   # Predicts the image using the trained model (outputs an array)
    prediction_single = prediction_single.argmax()          # Finds the biggest value in the array and output the index of that value
    prediction_single = CLASSES[prediction_single]          # Outputs the class name using the index

    # *** Plotting image with prediction
    fig = plt.figure(figsize = [10,10])
    fig.add_subplot(111)
    plt.imshow(img.reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted %s" % (prediction_single))
    plt.show()

    return prediction_single

# *** Use this function to input multiple images ==> outputs prediction
def predict_multiple_images(test_folder):
    images = []
    image_matrix = np.zeros((0,28, 28))

    # Loading images from test_folder
    for filename in os.listdir(test_folder):
        imgs = cv2.imread(os.path.join(test_folder,filename), 0)            # Reads the images in the folder and converts it to black and white
        if imgs is not None:
            images.append(imgs)                                             # Appends all the images in the folder (list of arrays)
    image_matrix = np.concatenate((image_matrix,images), axis=0)            # Converts the list of arrays into a matrix (to be inputted into predict_on_batch)
    prediction_multiple = model.predict_on_batch(image_matrix[:,:,:,None])  # Predicts the batch of images using the trained model (outputs a list of arrays)

    # Used to find how many images to place in each figure
    # n = len(images)             # computes number of input images
    # sqrt_n = math.sqrt(n)       # takes the square root
    # print(n)
    # if sqrt_n * sqrt_n == n:
    #     n = sqrt_n              # will ouput sqrt_n * sqrt_n images onto plot
    # else:
    #     n = int(sqrt(n)+1)      # finds next biggest square
    # print(n)
    #
    # # fig = plt.figure(figsize = [10,10])
    # # for i in range(len(prediction_multiple)):
    # #     prediction = prediction_multiple[i].argmax()        # Finds the biggest value in the array and output the index of that value
    # #     prediction = CLASSES[int(prediction)]               # Outputs the corresponding class name of array
    # #     print("Predicted %s" % (prediction))
    # #
    # #     # *** Plotting images with Prediction
    # #     fig.add_subplo172t(n,n,i+1)
    # #     plt.imshow(images[i], cmap='gray', interpolation='none')
    # #     plt.axis('off')
    # #     plt.title("Predicted %s" % (prediction))
    # #
    # # plt.show()

    return prediction

# *** Use this to input single images
# script_dir = sys.path[0]
# img_path = os.path.join(script_dir, folder_name + '/'+ image_number + '.png')
# prediction = predict_single_image(img_path)
# print("Predicted %s, Class %s" % (prediction,folder_name))


# predictions = predict_multiple_images(test_folder)
