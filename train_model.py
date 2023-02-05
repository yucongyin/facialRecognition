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
import os

def train():
    #test flag, 0 is test mode, 1 is live mode
    flag = 0

    lfw_people = fetch_lfw_people(data_home='./lfw-deepfunneled',min_faces_per_person=1, resize=1)
    print("load is done.")
    #the vectorized data
    X = lfw_people.data
    labels = lfw_people.target
    X = resize(X,(X.shape[0],1024,),anti_aliasing=True)
    for i in range(numpy.size(labels)):

        #X[i] = X[i].flatten()
        labels[i] = 1
    print("X[0]: ",X[1])
        
    if flag == 1:    
        #print("size:",numpy.size(labels))
        non_face_path = "./non_faces"
        for images in os.listdir(non_face_path):
            #print(images)
            image = skimage.io.imread(os.path.join(non_face_path,images),as_gray=True)
            image = resize(image,(32,32),anti_aliasing=True)
            image = image.flatten()
            X = numpy.vstack((X,image))
            print("X in appending:",X[0])
            #print("image:",image)
            labels = numpy.append(labels,0)
    else:


        with open('data.pkl', 'rb') as f:
            X = pickle.load(f)
        non_face_path = "./non_faces"
        for images in os.listdir(non_face_path):
            labels = numpy.append(labels,0)

    

    print("non-face done")
    print("after non-face X[0]: ",X[0])
    with open("data.pkl", "wb") as file:
        pickle.dump(X, file,protocol=pickle.HIGHEST_PROTOCOL)

    #data proprocessing
    indexes = list(range(numpy.size(labels)))
    random.shuffle(indexes)
    print(numpy.size(labels))
    X = X[indexes]
    labels = labels[indexes]
    #print("X[0]after shuffle: ",X[0])
    # Split the data into training and testing sets, here we choose a test_size of 0.1 which means 90% of the image is used for training and only 10% is for test 
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)
    

    # Initial definition of the MLP classifier
    #set up 3 hidden layer, each one has 150,100,50 layers, with an epoch of 1000 to make the model more accurate
    #uses solver = 'adam' because according to documentation it is more useful for big training data size
    #learning rate was initially 0.1
    #it reaches a 0.583 test score with 0.685 loss, in 133.71second, with early_stopping, not turn it off to see the limit, change learning rate to 0.001 
    model = MLPClassifier(hidden_layer_sizes=(50),shuffle=True, max_iter=1000, activation = 'relu',
                    solver='adam', verbose=10, tol=0.00000000, random_state=1,
                    learning_rate_init=0.1)
    t0 = time.time()
    model.fit(X_train, y_train)

    # Evaluate the performance of the model
    score = model.score(X_test, y_test)
    print("Training time:", time.time()-t0)
    print("Test score: {:.3f}".format(score))
    # Save the model to a file
    with open("model8.pkl", "wb") as f:
        pickle.dump(model, f)

    print("model saved")



train()