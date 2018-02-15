import segmentation
import cv2
import os

# Classes 
CLASSES = ['one','two','three','four','five','six','seven', 'eight','nine','plus','minus','multiplication', 'division1', 'division2', 'period']

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = os.path.join(folder,filename)
        if img is not None:
            images.append(img)
	print (images)
    return images

def segmentData(input_directory):
	print("5")
    symbols = load_images_from_folder(input_directory)

    for i in range(len(symbols)):
        images = segmentation.segment(symbols[i])

        for thing in range(len(CLASSES)):
            cv2.imwrite(thing + '/' + thing + '_' + str(i) + '.png', images[thing])

segmentData('images')
