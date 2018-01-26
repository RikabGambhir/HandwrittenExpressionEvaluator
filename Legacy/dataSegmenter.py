import segmentation
import cv2
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = os.path.join(folder,filename)
        if img is not None:
            images.append(img)
    return images

def segmentData(input_directory, classes):

    symbols = load_images_from_folder(input_directory)

    for i in range(len(symbols)):
        images = segmentation.segment(symbols[i])

        for class_ in range(len(classes)):
            cv2.imwrite(classes[class_] + '/' + classes[class_] + '_' + str(i) + '.png', images[class_])
