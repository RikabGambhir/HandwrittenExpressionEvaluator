import numpy as np
import cv2
import itertools
import os

#this is the cascade we just made. Call what you want


def haar_cascade(img, cascade):


    symbol_cascade = cv2.CascadeClassifier(cascade)
    print(img)
    symbols = symbol_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5, minSize=(10, 10), flags = cv2.CASCADE_SCALE_IMAGE)
    print(symbols)
    return combine_boxes(symbols)


def combine_boxes(boxes):
    new_array = []
    for boxa, boxb in itertools.combinations(boxes, 2):
        if intersection(boxa, boxb):
            new_array.append(union(boxa, boxb))
        else:
            new_array.append(boxa)

    to_remove = []
    for box_i in new_array:
        for box_j in new_array:
            if (np.array_equal(intersection(box_i, box_j), box_j )):
                to_remove.append(box_j)

    new_array  = [element for element in new_array if element not in to_remove]
    return np.array(new_array).astype('int')

def union(a,b):
  x = min(a[0], b[0])
  y = min(a[1], b[1])
  w = max(a[0]+a[2], b[0]+b[2]) - x
  h = max(a[1]+a[3], b[1]+b[3]) - y
  return (x, y, w, h)

def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return () # or (0,0,0,0) ?
  return (x, y, w, h)




# add this
# image, reject levels level weights.

# add this
# for (x,y,w,h) in symbols:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
#
#
# cv2.imwrite("test_fives_haar.jpg", img)
# print('Done')



# for file_type in ['negative']:
#
#     for img in os.listdir(file_type):
#
#
#         if file_type == 'negative':
#                 line = "./" + file_type+'/'+img+'\n'
#                 with open('bg.txt','a') as f:
#                     f.write(line)
