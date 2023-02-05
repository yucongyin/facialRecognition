import numpy
import scipy
import matplotlib.pyplot
import random
import math
import skimage.measure
import skimage.io
import skimage.viewer
from PyQt5 import QtCore, QtGui, QtWidgets

def convolve2d(image, kernel):
    # This function which takes an image and a kernel 
    # and returns the convolution of them
    # Args:
    #   image: a numpy array of size [image_height, image_width].
    #   kernel: a numpy array of size [kernel_height, kernel_width].
    # Returns:
    #   a numpy array of size [image_height, image_width] (convolution output).
    
    kernel = numpy.flipud(numpy.fliplr(kernel))    # Flip the kernel
    output = numpy.zeros_like(image)            # convolution output
    # Add zero padding to the input image
    image_padded = numpy.zeros((image.shape[0] + 2, image.shape[1] + 2))   
    image_padded[1:-1, 1:-1] = image
    for x in range(image.shape[1]):     # Loop over every pixel of the image
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y,x]=(kernel*image_padded[y:y+3,x:x+3]).sum()        
    return output


def convolve2dBig(image, kernel):
    # This function which takes an image and a kernel 
    # and returns the convolution of them
    # Args:
    #   image: a numpy array of size [image_height, image_width].
    #   kernel: a numpy array of size [kernel_height, kernel_width].
    # Returns:
    #   a numpy array of size [image_height, image_width] (convolution output).
    
    kernel = numpy.flipud(numpy.fliplr(kernel))    # Flip the kernel
    output = numpy.zeros_like(image)            # convolution output
    # Add zero padding to the input image
    image_padded = numpy.zeros((image.shape[0] + 8, image.shape[1] + 8))   
    image_padded[4:-4, 4:-4] = image
    for x in range(image.shape[1]):     # Loop over every pixel of the image
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y,x]=(kernel*image_padded[y:y+9,x:x+9]).sum()        
    return output


# CNN implementation
img = skimage.io.imread('image_small.png', as_gray=True)  # load the image as grayscale
W = numpy.random.random(img.size,)
print ("image matrix size: ", img.shape)      # print the size of image
print ("\n First 5 columns and rows of the image matrix: \n", img[:5,:5]*255) 
skimage.viewer.ImageViewer(img).show()              # plot the image

# Convolve the sharpen kernel (filter) and the image
sharpen_kernel = numpy.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
edge_kernel = numpy.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
blur_kernel = numpy.array([[1,1,1],[1,1,1],[1,1,1]])/9.0

#bigKernel = numpy.random.random([9,9])
#I = convolve2dBig(img,bigKernel)
#I = I / numpy.max(I)

I = convolve2d(img,edge_kernel)
skimage.viewer.ImageViewer(I).show()

#I = scipy.signal.convolve2d(img, edge_kernel, 'same')
print ("\n First 5 columns and rows of the matrix: \n", I[:5,:5]*255)

# relu
I = numpy.where(I>0,I,0)

# hidden layer at output
z = 1/(1 + math.exp(-numpy.dot(I.reshape(-1,),W)))

Ipool = skimage.measure.block_reduce(I,(2,2),numpy.max)

# Plot the filtered image
skimage.viewer.ImageViewer(Ipool).show()


print("done")



