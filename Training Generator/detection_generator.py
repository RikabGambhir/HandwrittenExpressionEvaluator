import os
import os.path
import cv2
import numpy as np
import random


# Possible additions include methods to evaluate expressions and generate
# some random expressions.


CLASSES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'plus', 'minus', 'multiplication', 'division1']
SYMBOLS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/']


def load_random_from_folder(folder):

    a=random.choice(os.listdir(folder))
    # print(a)

    img = cv2.imread(os.path.join(folder,a),0) # Reads the images from folder
    return img

class Expression:
    pass

class Number(Expression):
    def __init__(self, num):
        self.num = num

    def __str__(self):
        return str(self.num)

class BinaryExpression(Expression):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def __str__(self):
        return str(self.left) + " " + self.op + " "  + str(self.right)

def randomExpression(prob):
    p = np.random.rand()
    if p > prob:
        return Number(np.random.randint(2, 100))
    elif np.random.randint(2) == 0:
        return randomExpression(prob / 1.2)
    else:
        left = randomExpression(prob / 1.2)
        op = random.choice(["+", "-", "*", "/"])
        right = randomExpression(prob / 1.2)
        return BinaryExpression(left, op, right)


def generate(n):
    for i in range(0,n):
        start = np.zeros((28, 1))
        expression = randomExpression(1)
        print(expression)
        for char in str(expression):
            print(char)
            if char != ' ':
                index = SYMBOLS.index(char)
                img = load_random_from_folder('../Data/' + CLASSES[index])
                start  = np.concatenate((start, img), axis = 1)
        cv2.imwrite(str(expression) + ".png", start)

generate(10)
