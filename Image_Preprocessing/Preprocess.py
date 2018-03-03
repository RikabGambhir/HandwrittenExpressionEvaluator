import cv2
import numpy as np
from matplotlib import pyplot as plt


def process(img):
    img = cv2.GaussianBlur(img,(5,5),0)
    img = cv2.GaussianBlur(img,(5,5),0)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,7,2)
    img = 255 - img
    return img


