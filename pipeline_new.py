import cv2
import numpy as np
from Image_Preprocessing import Preprocess
from Classification import classifier
import os
import matplotlib.pyplot as plt
from HarrCascade import haar_cascade
from Evaluation import BoundingBox
from Evaluation import DetectionOutput


DIRECTORY = 'Data/test'
CASCADE = 'HarrCascade/data/cascade.xml'

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


    proc = images[file_name]
    proc = Preprocess.image_resize(proc, width = 250)

    proc = Preprocess.process(proc)
    # proc = Preprocess.crop(proc, 8, 128)
    #
    fig = plt.figure(figsize = [10,10])
    fig.add_subplot(131)
    plt.imshow(images[file_name], cmap='gray', interpolation='none')
    plt.title("Before Preprocessing" )
    # fig.add_subplot(132)
    # plt.imshow(proc1,cmap='gray', interpolation = 'none')
    fig.add_subplot(133)
    plt.imshow(proc, cmap='gray', interpolation='none')
    #
    #


    proc2 = proc.copy()
    proc3 = proc.copy()

    symbols, s2 = haar_cascade.haar_cascade(proc, CASCADE)
    boundingBoxes = []
    for (x,y,w,h) in symbols:
        # cv2.rectangle(proc2,(x,y),(x+w,y+h),(255,255,0),2)

        crop_img = proc[y:y+h, x:x+w]
        crop_img = Preprocess.resizeAndPad(crop_img, (28, 28))
        pred = classifier.predict_single_image(crop_img)
        print('{:<30}{:<15}'.format(file_name, pred))

        box = BoundingBox.BoundingBox(x,y, x+w, y+h, classifier.get_symbol(pred))
        boundingBoxes.append(box)
        cv2.imwrite("test_fives_haar.jpg", proc)



    for (x,y,w,h) in symbols:
        cv2.rectangle(proc2,(x,y),(x+w,y+h),(255,255,0),2)
    for (x,y,w,h) in s2:
        cv2.rectangle(proc3,(x,y),(x+w,y+h),(255,255,0),2)


    fig.add_subplot(132)
    plt.imshow(proc3, cmap='gray', interpolation='none')
    fig.add_subplot(133)
    plt.imshow(proc2, cmap='gray', interpolation='none')


    detectionOutput = DetectionOutput.DetectionOutput(boundingBoxes)
    string = detectionOutput.combineAll().display()
    print(string)

    ans = ""
    try:
        ans = string + " = " + str(eval(string))
    except:
        ans = string + " = ???"

    print("***** FINAL ANSWER *****")
    try:
        print(string + " = " + str(eval(string)))
    except:
        print(string + " = ???")
    print("***** DONE *****")

    plt.title(ans)

    plt.show()


    cv2.imwrite("test_fives_haar.jpg", proc)
    print('Done')






    # pred = classifier.predict_single_image(proc)
    # print('{:<30}{:<15}'.format(file_name, pred))
    # path = file_name
    # os.remove(os.path.abspath(path))
    # labels.append(pred)
