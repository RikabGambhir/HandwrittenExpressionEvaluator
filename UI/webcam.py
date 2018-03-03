import cv2

cam = cv2.VideoCapture(0)
s, im = cam.read() # captures image
# cv2.imshow("Test Picture", im) # displays captured image
cv2.imwrite("test.bmp",im) # writes image test.bmp to disk
