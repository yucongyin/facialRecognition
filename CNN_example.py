import numpy as np

# defining the Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

# derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

images = np.load('faces.npy')
images2 = np.load('non_faces.npy')
faces = images[:,:,0:100]
non_faces = images2[:,:,0:100]

X = np.concatenate((faces,non_faces), axis = 2)
Xlabels = np.concatenate((np.ones(100), np.zeros(100)))

# set the learning rate
lr = 0.01

# initializing filter
f=np.random.uniform(size=(3,5,5))
f = f.T

#print('Filter 1', '\n', f[:,:,0], '\n')
#print('Filter 2', '\n', f[:,:,1], '\n')
#print('Filter 3', '\n', f[:,:,2], '\n')

# Generating patches from images
new_image = []

print("creating all patches for all images...")
# for number of images
for k in range(X.shape[2]):
    print("image ",k,"       ",end = '\r', flush = True)
    # sliding in horizontal direction
    for i in range(X.shape[0]-f.shape[0]+1):
        # sliding in vertical direction
        for j in range(X.shape[1]-f.shape[1]+1):
            new_image.append(X[:,:,k][i:i+f.shape[0],j:j+f.shape[1]])
            
# resizing the generated patches as per number of images
new_image = np.array(new_image)
new_image.resize((X.shape[2],int(new_image.shape[0]/X.shape[2]),new_image.shape[1],new_image.shape[2]))
new_image.shape

# number of features in data set
s_row = X.shape[0] - f.shape[0] + 1
s_col = X.shape[1] - f.shape[1] + 1
num_filter = f.shape[2]

inputlayer_neurons = (s_row)*(s_col)*(num_filter)
output_neurons = 1 

# initializing weight for hidden layer fully connected neurons
wo=np.random.uniform(size=(inputlayer_neurons,output_neurons))

epochError = np.zeros(100)

print("...............................")
for epoch in range(100):

    print("epoch ", epoch)

    # generating output of convolution layer
    filter_output = []

    print("forward calculation of convolution...")
    # for each image
    for i in range(len(new_image)):
        print("image ",i,"       ",end = '\r', flush = True)
        # apply each filter
        for k in range(f.shape[2]):
            # do element wise multiplication
            for j in range(new_image.shape[1]):
                filter_output.append((new_image[i][j]*f[:,:,k]).sum()) 

    filter_output = np.resize(np.array(filter_output), (len(new_image),f.shape[2],new_image.shape[1]))

    # applying activation over convolution output
    filter_output_sigmoid = sigmoid(filter_output)

    #filter_output.shape, filter_output_sigmoid.shape

    # generating input for fully connected layer
    filter_output_sigmoid = filter_output_sigmoid.reshape((filter_output_sigmoid.shape[0],filter_output_sigmoid.shape[1]*filter_output_sigmoid.shape[2]))
    filter_output_sigmoid = filter_output_sigmoid.T

    # Linear trasnformation for fully Connected Layer
    output_layer_input= np.dot(wo.T,filter_output_sigmoid)
    output_layer_input = (output_layer_input - np.average(output_layer_input))/np.std(output_layer_input)

    # activation function
    output = sigmoid(output_layer_input)

    epochError[epoch] = np.sum(Xlabels != np.round(output,0))
    print("............................................")
    print("total errors in epoch ",epoch," are ", epochError[epoch])
    print("............................................")

    #Error
    error = np.square(Xlabels-output)/2

    #Change in error w.r.t output (Gradient)
    error_wrt_output = -(Xlabels-output)

    #Change in error w.r.t sigmoid transformation z2 (output_layer_input)
    output_wrt_output_layer_input=output*(1-output)

    #Change in z2 w.r.t weights
    output_wrt_w=filter_output_sigmoid


    #delta change in w for fully connected layer
    delta_error_fcp = np.dot(output_wrt_w,(error_wrt_output * output_wrt_output_layer_input).T)

    print("update the weights in the hidden layer")
    #update the weights in the fully connected layer
    wo = wo - lr*delta_error_fcp

    #Change in z2 w.r.t sigmoid output A1
    output_layer_input_wrt_filter_output_sigmoid = wo.T

    #Change in A1 w.r.t sigmoid transformation z1
    filter_output_sigmoid_wrt_filter_output = filter_output_sigmoid * (1-filter_output_sigmoid)

    # calculating derivatives for backprop convolution
    # dE/D0 * dO/dz2 * dz2/dA1 * dA1/dz1 
    error_wrt_filter_output = np.dot(output_layer_input_wrt_filter_output_sigmoid.T,error_wrt_output*output_wrt_output_layer_input) * filter_output_sigmoid_wrt_filter_output
    error_wrt_filter_output = np.average(error_wrt_filter_output, axis=1)
    # resize to filter sizes
    error_wrt_filter_output = np.resize(error_wrt_filter_output,(X.shape[0]-f.shape[0]+1,X.shape[1]-f.shape[1]+1, f.shape[2]))


    # convolve dz1/df (which is the input X) with the rest of the derivatives
    print("back propagation convolution")
    filter_update = []
    for i in range(f.shape[2]):
        for j in range(f.shape[0]):
            for k in range(f.shape[1]):            
                temp = 0
                spos_row = j
                spos_col = k
                epos_row = spos_row + s_row
                epos_col = spos_col + s_col
                for l in range(X.shape[2]):
                    temp = temp + (X[spos_row:epos_row,spos_col:epos_col,l]*error_wrt_filter_output[:,:,i]).sum()
                filter_update.append(temp/X.shape[2])  

    filter_update_array = np.array(filter_update)
    filter_update_array = np.resize(filter_update_array,(f.shape[2],f.shape[0],f.shape[1]))

    print("update the filters")
    # update the filters using the learning rate
    for i in range(f.shape[2]):
        f[:,:,i] = f[:,:,i] - lr*filter_update_array[i]

print("done")