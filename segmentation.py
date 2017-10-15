import cv2
import numpy as np


def is_Black(rows,col,img):
    return (np.all(img[:,col] <= 28*[50]))


def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img


def segment(imageFile):

    img = cv2.imread(imageFile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32')

    segments = []
    i = 0
    rows, columns = img.shape
    leftIndex = -1
    rightIndex = -1
    for column in range(columns):
        isBlack = True;

        if (leftIndex > -1 and rightIndex == -1):
            if is_Black(rows, column, img):
                rightIndex = column

        for row in range(rows):
            if (leftIndex == -1):
                if img[row,column] >= 128:
                    leftIndex = column


        if (rightIndex > leftIndex):

            width = rightIndex - leftIndex + 1
            columsToFill = rows - width
            digit = img[:,leftIndex:rightIndex]
            # Pad the image on the left and right with 0's
            digit = resizeAndPad(digit, (rows,rows), padColor = 0)
            cv2.imwrite(str(i) + ".png", digit)
            segments.append(digit)
            i = i + 1
            leftIndex = -1
            rightIndex = -1

    cv2.imwrite('image1.png', segments[0])

    return segments

# segment("test7.png")
