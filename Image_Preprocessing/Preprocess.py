import cv2
import numpy as np
from matplotlib import pyplot as plt


def process(img):
    img = cv2.GaussianBlur(img,(5,5),0)
    img = cv2.GaussianBlur(img,(5,5),0)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,15,2)
    img = cv2.fastNlMeansDenoising(img, h = 3)
    img = 255 - img
    return img

def crop(img, marginval, threshold):
        shape = np.shape(img)
        top=shape[1]
        bottom=0
        left=shape[0]
        right=0
        for rows in range(0,shape[0]):
            for columns in range(0,shape[1]):
                if img[rows][columns] > threshold:
                    if rows<top:
                        top = rows #low number
                    if rows>bottom:
                        bottom=rows #high
                    if columns<left:
                        left=columns#low
                    if columns>right:
                        right=columns#high
        top -= marginval #add whitespace later
        bottom += marginval
        left -= marginval
        right += marginval
        cropped = img[top:bottom+1][left:right+1]
        return cropped

#
# image = cv2.imread("test_one.png", 0)
# image = process(image)
# cv2.imwrite('test.png', image)
# crop(image, 1, 128)
