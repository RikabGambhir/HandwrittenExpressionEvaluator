import cv2
import numpy as np
from matplotlib import pyplot as plt

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def process(img):
    img = cv2.fastNlMeansDenoising(img, h = 3)

    img = cv2.GaussianBlur(img,(25,11),0)
    # img = cv2.GaussianBlur(img,(11,11),0)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,25,2)
    # img = cv2.fastNlMeansDenoising(img, h = 3)
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
