import numpy
import tensorflow as tf
import sys
from PIL import Image
import os
from numpy import asarray


numpy.set_printoptions(threshold=sys.maxsize)

resolution = (640,360)

def data():
    x_train = []
    y_train = []

    directory = r'C:\Users\Max\Desktop\ai\destiny 2\traindata\training\enemies' #enemies - 1
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)

        if os.path.isfile(f):
            image = Image.open(f)
            image = image.resize(resolution)
            image = asarray(image)
            x_train.append(image)

            y_train.append(1)


    directory = r'C:\Users\Max\Desktop\ai\destiny 2\traindata\training\noenemies' #no ememies - 0
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)

        if os.path.isfile(f):
            image = Image.open(f)
            image = image.resize(resolution)
            image = asarray(image)
            x_train.append(image)

            y_train.append(0)

    return x_train,y_train

def testdata():
    x_test = []
    y_test = []

    directory = r'C:\Users\Max\Desktop\ai\destiny 2\traindata\testing\enemies' #enemies - 1
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)

        if os.path.isfile(f):
            image = Image.open(f)
            image = image.resize(resolution)
            image = asarray(image)
            x_test.append(image)

            y_test.append(1)


    directory = r'C:\Users\Max\Desktop\ai\destiny 2\traindata\testing\no enemies' #no ememies - 0
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)

        if os.path.isfile(f):
            image = Image.open(f)
            image = image.resize(resolution)
            image = asarray(image)
            x_test.append(image)

            y_test.append(0)

    return x_test,y_test

