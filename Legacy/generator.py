import cv2
import dataSegmenter
import dataaugmentation
import os

symbols = ("0","1","2","3","4","5","6","7","8","9","+","-","*","","^",".","(",")","=")
input_directory = "samples"

for i in range(len(symbols)):
    if not os.path.exists(str(i)):
        os.makedirs(str(i))

dataSegmenter.segmentData(input_directory, symbols)
dataaugmentation.generate(symbols)
