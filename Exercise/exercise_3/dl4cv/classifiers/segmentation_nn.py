"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision

class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        model_conv = torchvision.models.resnet18(pretrained=True)
        for param in model_conv.parameters():
            param.requires_grad = False
        
        # Parameters of newly constructed modules have requires_grad=True by default
        
        self.model_conv = nn.Sequential(
            model_conv.conv1,
            model_conv.bn1,
            model_conv.relu,
            model_conv.maxpool,
            model_conv.layer1,
            model_conv.layer2,
            model_conv.layer3,
            model_conv.layer4,
        )
        
        self.conv1 = model_conv.conv1
        self.bn1 = model_conv.bn1
        self.relu = model_conv.relu
        self.maxpool = model_conv.maxpool
        self.layer1 = model_conv.layer1
        self.layer2 = model_conv.layer2
        self.layer3 = model_conv.layer3
        self.layer4 = model_conv.layer4
        
        
        
        
        self.my_model = nn.Sequential(
            nn.Conv2d(512, 100, kernel_size=3, stride=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(100, 70, kernel_size = 7, stride = 5, padding = (1,1)),
            nn.BatchNorm2d(70, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(70, 50, kernel_size = 8, stride = 3, padding = (1,1)),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True),
            nn.ConvTranspose2d(50, 24, kernel_size = 7, stride = 5, padding = (1,1)),
        )

        self.conv2d = nn.Conv2d(512, 100, kernel_size=3, stride=3, padding=1)
        self.nnrelu = nn.ReLU()
        self.convtranspose2d = nn.ConvTranspose2d(100, 70, kernel_size = 7, stride = 5, padding = (1,1))
        self.batchnorm2d = nn.BatchNorm2d(70, eps=1e-05, momentum=0.1, affine=True)
        
        self.convtranspose2d_2 = nn.ConvTranspose2d(70, 50, kernel_size = 8, stride = 3, padding = (1,1))
        self.batchnorm2d_2 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True)
        self.convtranspose2d_3 = nn.ConvTranspose2d(50, 24, kernel_size = 7, stride = 5, padding = (1,1))

        
        
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        #out = self.model_conv(x)
        #out = self.my_model(out)
        
        #print("original shape: ", x.size())
        x = self.conv1(x)
        #print("after conv1 shape: ", x.size())
        x = self.bn1(x)
        #print("after bn1 shape: ", x.size())
        x = self.relu(x)
        #print("after relu shape: ", x.size())
        x = self.maxpool(x)
        #print("after maxpool shape: ", x.size())
        x = self.layer1(x)
        #print("after layer1 shape: ", x.size())
        x = self.layer2(x)
        #print("after layer2 shape: ", x.size())
        x = self.layer3(x)
        #print("after layer3 shape: ", x.size())
        x = self.layer4(x)
        #print("final, after layer4 shape: ", x.size())
        
        
        #print("before my model sequence shape: ", x.size())
        x = self.conv2d(x)
        #print("after 1st con2d shape: ", x.size())
        x = self.relu(x)
        #print("after 1st relu shape: ", x.size())
        x = self.convtranspose2d(x)
        #print("after 1st convtranspose2d shape: ", x.size())
        x = self.batchnorm2d(x)
        #print("after 1st batchnorm2d shape: ", x.size())
        x = self.relu(x)
        #print("after 2nd relu shape: ", x.size())
        x = self.convtranspose2d_2(x)
        #print("after 2nd convtranspose2d shape: ", x.size())
        x = self.batchnorm2d_2(x)
        #print("after 2nd batchnorm2d shape: ", x.size())
        x = self.convtranspose2d_3(x)
        #print("final and after 3rd convtranspose2d shape: ", x.size())
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

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
