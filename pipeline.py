import cv2
import numpy as np
from Image_Preprocessing import Preprocess
from Classification import classifier
import os

def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),0) # Reads the images from folder
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # Converts images to gray scale
        if img is not None:
            images[os.path.join(folder,filename)] = img
    return images


while True:

    images = load_images_from_folder("Data/images")
    print(images)
    for file_name in images.keys():
        proc = Preprocess.process(images[file_name])
        pred = classifier.predict_single_image(proc)
        path = file_name
        os.remove(os.path.abspath(path))
