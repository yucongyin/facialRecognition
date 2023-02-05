import pickle
import time
import numpy
import scipy
import matplotlib.pyplot
import random
import math
import skimage.measure
import skimage.io
import skimage.viewer
from skimage.transform import resize
from sklearn import svm
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from PyQt5 import QtCore, QtGui, QtWidgets
import os
   
def test():   
    lfw_people = fetch_lfw_people(data_home='./lfw-deepfunneled',min_faces_per_person=1, resize=1)
    print("load is done.")
    #the vectorized data
    X = lfw_people.data
    labels = lfw_people.target
    X = resize(X,(X.shape[0],1024,),anti_aliasing=True)
    print(X[0])  



test()