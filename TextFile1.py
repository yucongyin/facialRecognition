# learning rate
eta=.001
for epoch in range(5):
    # custom code to keep track of quantities to 
    # keep a running average. it is not shown for clarity. 
    # the reader can implement her own or ask me in the comments.
    train_loss, train accuracy=averager(), averager()
    
    for i in range(len(y_train)):
        
        # Take a random sample from train set
        k=np.random.randint(len(y_train))
        X=X_train[k]
        y=y_train[k]

        ##### FORWARD PASS ######
        # First layer is just the input
        l0=X
        
        # Embed the image in a bigger image. 
        # It would be useful in computing corrections 
        # to the convolution filter
        lt0=np.zeros((l0.shape[0]+K-1,l0.shape[1]+K-1))
        lt0[K//2:-K//2+1,K//2:-K//2+1]=l0
        
        # convolve with the filter
        # Layer one is Relu applied on the convolution        
        l0_conv=convolve(l0,W1[::-1,::-1],'same','direct')
        l1=relu(l0_conv)
        # Compute layer 2
        l2=sigmoid(np.dot(l1.reshape(-1,),W2))
        l2=l2.clip(10**-16,1-10**-16)
        
        ####### LOSS AND ACCURACY #######
        loss=-(y*np.log(l2)+(1-y)*np.log(1-l2))
        accuracy=int(y==np.where(l2>0.5,1,0))
        
        # Save the loss and accuracy to a running averager
        train_loss.send(loss)
        train_accuracy.send(accuracy)
        ##### BACKPROPAGATION #######
        
        # Derivative of loss wrt the dense layer
        dW2=(((1-y)*l2-y*(1-l2))*l1).reshape(-1,)
        
        # Derivative of loss wrt the output of the first layer
        dl1=(((1-y)*l2-y*(1-l2))*W2).reshape(28,28)
        
        # Derivative of the loss wrt the convolution filter
        f1p=relu_prime(l0_conv)
        dl1_f1p=dl1*f1p
        dW1=np.array([[
           (lt0[alpha:+alpha+image_size,beta:beta+image_size]\
           *dl1_f1p).sum() for beta in range(K)
        ]for alpha in range(K)])
        W2+=-eta*dW2
        W1+=-eta*dW1
    loss_averager_valid=averager()
    accuracy_averager_valid=averager()   
    
    for X,y in zip(X_valid,y_valid):
        accuracy,loss=forward_pass(W1,W2,X,y)
        loss_averager_valid.send(loss)
        accuracy_averager_valid.send(accuracy)
    
    train_loss,train_accuracy,valid_loss,valid_accuracy\
            =map(extract_averager_value,[train_loss,train_accuracy,
                 loss_averager_valid,accuracy_averager_valid])
    
    # code to print losses and accuracies suppressed for clarity