"""ClassificationCNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationCNN(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, kernel_size=7,
                 stride_conv=1, weight_scale=0.001, pool=2, stride_pool=2, hidden_dim=100,
                 num_classes=10, dropout=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - filter_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride_conv: Stride for the convolution layer.
        - stride_pool: Stride for the max pooling layer.
        - weight_scale: Scale for the convolution weights initialization
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(ClassificationCNN, self).__init__()
        channels, height, width = input_dim

        ########################################################################
        # TODO: Initialize the necessary trainable layers to resemble the      #
        # ClassificationCNN architecture  from the class docstring.            #
        #                                                                      #
        # In- and output features should not be hard coded which demands some  #
        # calculations especially for the input of the first fully             #
        # convolutional layer.                                                 #
        #                                                                      #
        # The convolution should use "same" padding which can be derived from  #
        # the kernel size and its weights should be scaled. Layers should have #
        # a bias if possible.                                                  #
        #                                                                      #
        # Note: Avoid using any of PyTorch's random functions or your output   #
        # will not coincide with the Jupyter notebook cell.                    #
        ########################################################################
        #conv - relu - 2x2 max pool - fc - dropout - relu - fc
        padding = int((kernel_size - 1)/2)
        Hh = 1 + int((height + 2 * padding - kernel_size) / stride_conv)
        Hw = 1 + int((width + 2 * padding - kernel_size) / stride_conv)
        
        H1 = int((Hh - pool) / stride_pool) + 1
        W1 = int((Hw - pool) / stride_pool) + 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(channels, num_filters, kernel_size=kernel_size, padding=padding, stride=stride_conv), #c*h*w => num_filters*(h-2)*(w-2)
            nn.ReLU(),
            nn.MaxPool2d(pool, stride=stride_pool), #num_filters*(h-2)/2*(w-2)/2 n+2.p-f/2 + 1  ((height + 2*2 - num_filters)/stride_conv) + 1
            
            nn.Linear(num_filters*H1*W1, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU()
        )
        
        self.conv2d = nn.Conv2d(channels, num_filters, kernel_size=kernel_size, padding=padding, stride=stride_conv)
        self.relu = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(pool, stride=stride_pool)
        self.fc = nn.Linear(num_filters*H1*W1, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        
        
        
        self.out = nn.Linear(hidden_dim, num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x):
        """
        Forward pass of the neural network. Should not be called manually but by
        calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        ########################################################################
        # TODO: Chain our previously initialized fully-connected neural        #
        # network layers to resemble the architecture drafted in the class     #
        # docstring. Have a look at the Variable.view function to make the     #
        # transition from the spatial input image to the flat fully connected  #
        # layers.                                                              #
        ########################################################################
        
        #x = self.layer1(x)
        #print("original shape: ", x.size())
        x = self.conv2d(x)
        #print("shape after conv2d: ", x.size())
        x = self.relu(x)
        #print("shape after first relu: ",x.size())

        x = self.maxpool2d(x)
        #print("shape after maxpool2d: ", x.size())
        x = x.view(x.size(0), -1) 
        #print("shape after flatenning the first relu output: ",x.size())
        x = self.fc(x)
        #print("shape after first fc: ", x.size())
        x = self.dropout(x)
        #print("shape after dropout: ", x.size())
        x = self.relu(x)
        #print("shape after second relu: ",x.size())
        x = x.view(x.size(0), -1) 
        #print("shape after flatenning the second relu output: ",x.size())
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        #print("final fc output shape: ", output.size())
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return output

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
