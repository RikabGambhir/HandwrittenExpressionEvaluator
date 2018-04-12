import cv2
import numpy as np
from Image_Preprocessing import Preprocess
from classification_folder import classifier
import os

DIRECTORY = 'Data/all_images'

def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),0) # Reads the images from folder
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # Converts images to gray scale
        if img is not None:
            images[os.path.join(folder,filename)] = img
    return images


#while True:

images = load_images_from_folder(DIRECTORY)
labels = []

print('{:<30}- {:<15}'.format('File', 'Prediction'))

# Classify each image in directory and display file name and prediction
for file_name in images.keys():
    #proc = images[file_name]
    proc = Preprocess.process(images[file_name])
    pred = classifier.predict_single_image(proc)
    print('{:<30}{:<15}'.format(file_name, pred))
    path = file_name
    # os.remove(os.path.abspath(path))
    labels.append(pred)
