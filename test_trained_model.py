import pickle
import time
import skimage.measure
import skimage.io
import skimage.viewer
import skimage.draw
from skimage.transform import resize
import cv2
import PyQt5
# Load the trained MLP model and the image
with open('model7.pkl', 'rb') as f:
    model = pickle.load(f)

#image = skimage.io.imread('./lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg',as_gray=True)
image = skimage.io.imread('test.jpg',as_gray=True)
#image = resize(image,(200,400),anti_aliasing=True)
#image = image.flatten()

# Set the window size and stride
window_size = (32, 32)
stride = (32, 32)
positive = 0
negative = 0
t0 = time.time()
# Slide the window over the image and use the model to predict whether each window contains a face
for y in range(0, image.shape[0] - window_size[1] , stride[1]):
    for x in range(0, image.shape[1] - window_size[0], stride[0]):
        window = image[y:y + window_size[1], x:x + window_size[0]]
        window = window.reshape((1, -1))
        prediction = model.predict(window)
        
        # If the classifier predicts that the window contains a face, draw a rectangle around it
        if prediction[0] > 0.5:
            positive += 1
            cv2.rectangle(image, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)
        else:
            negative += 1
print("time costs: ", time.time()-t0)
# Show the image with the predicted face locations
skimage.io.imshow(image)
skimage.io.show()
print("positive: ",positive, " negative: ", negative)

