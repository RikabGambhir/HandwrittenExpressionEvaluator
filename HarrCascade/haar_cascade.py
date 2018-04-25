import numpy as np
import cv2
import itertools
import os

#this is the cascade we just made. Call what you want


def haar_cascade(img, cascade):


    symbol_cascade = cv2.CascadeClassifier(cascade)
    symbols = symbol_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=2, flags = cv2.CASCADE_SCALE_IMAGE)
    # symbols -= (-3, -3, 6, 6)
    return combine_boxes2(symbols), symbols



def combine_boxes2(boxes):
     noIntersectLoop = False
     noIntersectMain = False
     posIndex = 0
     # keep looping until we have completed a full pass over each rectangle
     # and checked it does not overlap with any other rectangle
     while noIntersectMain == False:
         noIntersectMain = True
         posIndex = 0
         # start with the first rectangle in the list, once the first
         # rectangle has been unioned with every other rectangle,
         # repeat for the second until done
         while posIndex < len(boxes):
             noIntersectLoop = False
             while noIntersectLoop == False and len(boxes) > 1:
                a = boxes[posIndex]
                listBoxes = np.delete(boxes, posIndex, 0)
                index = 0
                for b in listBoxes:
                    #if there is an intersection, the boxes overlap
                    if intersection(a + (-3, -3, 6, 6), b + (-3, -3, 6, 6)):
                        newBox = union(a,b)
                        listBoxes[index] = newBox
                        boxes = listBoxes
                        noIntersectLoop = False
                        noIntersectMain = False
                        index = index + 1
                        break
                    noIntersectLoop = True
                    index = index + 1
             posIndex = posIndex + 1

     return boxes.astype("int")


def combine_boxes(boxes):
    new_array = []
    for boxa, boxb in itertools.combinations(boxes, 2):
        if intersection(boxa, boxb):
            new_array.append(union(boxa, boxb))
        else:
            new_array.append(boxa)

    boxes = new_array
    new_array = []
    for boxa, boxb in itertools.combinations(boxes, 2):
        if intersection(boxa, boxb):
            new_array.append(union(boxa, boxb))
        else:
            new_array.append(boxa)
            new_array.append(boxb)


    to_remove = []
    for box_i in new_array:
        for box_j in new_array:
            if box_area(intersection(box_i, box_j)) >= 0.9 * box_area(box_j):
                # print(box_area(intersection(box_i, box_j)), box_area(box_j))

                if box_area(box_i) >= box_area(box_j):
                    removearray(new_array, box_j)



    return np.array(new_array).astype('int')


def non_max_suppression_fast(boxes, overlapThresh=1):
   # if there are no boxes, return an empty list
   if len(boxes) == 0:
      return []

   # if the bounding boxes integers, convert them to floats --
   # this is important since we'll be doing a bunch of divisions
   if boxes.dtype.kind == "i":
      boxes = boxes.astype("float")
#
   # initialize the list of picked indexes
   pick = []

   # grab the coordinates of the bounding boxes
   x1 = boxes[:,0]
   y1 = boxes[:,1]
   x2 = boxes[:,2]
   y2 = boxes[:,3]

   # compute the area of the bounding boxes and sort the bounding
   # boxes by the bottom-right y-coordinate of the bounding box
   area = (x2 - x1 + 1) * (y2 - y1 + 1)
   idxs = np.argsort(y2)

   # keep looping while some indexes still remain in the indexes
   # list
   while len(idxs) > 0:
      # grab the last index in the indexes list and add the
      # index value to the list of picked indexes
      last = len(idxs) - 1
      i = idxs[last]
      pick.append(i)

      # find the largest (x, y) coordinates for the start of
      # the bounding box and the smallest (x, y) coordinates
      # for the end of the bounding box
      xx1 = np.maximum(x1[i], x1[idxs[:last]])
      yy1 = np.maximum(y1[i], y1[idxs[:last]])
      xx2 = np.minimum(x2[i], x2[idxs[:last]])
      yy2 = np.minimum(y2[i], y2[idxs[:last]])

      # compute the width and height of the bounding box
      w = np.maximum(0, xx2 - xx1 + 1)
      h = np.maximum(0, yy2 - yy1 + 1)

      # compute the ratio of overlap
      overlap = (w * h) / area[idxs[:last]]

      # delete all indexes from the index list that have
      idxs = np.delete(idxs, np.concatenate(([last],
         np.where(overlap == 1)[0])))

   # return only the bounding boxes that were picked using the
   # integer data type
   return boxes[pick].astype("int")

def box_area(box):
    try:
        return box[2] * box[3]
    except:
        return 0

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

def removearray(L,arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')



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
