import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
 
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    
    # Get shapes
    
    train = X.shape[0]
    num_classes = W.shape[1]
    W = W.transpose(1,0)
    X = X.transpose(1,0)
    
    dW = np.zeros_like(W)
    
    for i in range(train):
        f_i = W.dot(X[:, i])
        #log_c = np.max(f_i)
        #f_i -= log_c #to eradicate numerical instability issue, but not required here as feature values are really small.
        
        sum_i = 0.0  
        for f_i_j in f_i:
            sum_i += np.exp(f_i_j)
       
        
        loss += -f_i[y[i]] + np.log(sum_i)
        
        for j in range(num_classes):
            p = np.exp(f_i[j])/sum_i
            dW[j, :] += (p-(j==y[i])) * X[:, i]
            
        loss /= train
        dW /= train
    
    loss += 0.5 * reg * np.sum(W * W)   
    dW += reg * W
    dW = dW.transpose(1,0)

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg, num_iters=0):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    train = X.shape[0]
    num_classes = W.shape[1]
    W = W.transpose(1,0)
    X = X.transpose(1,0)
    
    dW = np.zeros_like(W)
    
    for i in range(train):
        f_i = W.dot(X[:, i])
        log_c = np.max(f_i)
        f_i -= log_c 
        
        sum_i = 0.0  
        for f_i_j in f_i:
            sum_i += np.exp(f_i_j)
            
        loss += -f_i[y[i]] + np.log(sum_i)
        
        for j in range(num_classes):
            p = np.exp(f_i[j])/sum_i
            dW[j, :] += (p-(j==y[i])) * X[:, i]
            
        loss /= train
        dW /= train
    
    loss += 0.5 * reg * np.sum(W * W)   
    dW += reg * W
    dW = dW.transpose(1,0)

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

