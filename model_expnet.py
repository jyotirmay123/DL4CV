# ExpNet Class Implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class ExpNet(nn.Module):
    
    def __init__(self, useCuda, gpuDevice=0):
        
        super(ExpNet, self).__init__()

        self.gpuDevice = gpuDevice
        
        #{1st phase}: (conv - relu - max pool) * 5 - conv 

        self.conv2d1 = nn.Conv2d(3, 64, (3,3), (1,1), (1,1))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d((3,3), stride=(2,2), padding=(1,1))
        
        self.conv2d2 = nn.Conv2d(64, 128, (3,3), (1,1), (1,1))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d((3,3), stride=(2,2), padding=(1,1))
        
        self.conv2d3 = nn.Conv2d(128, 256, (3,3), (1,1), (1,1))
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d((3,3), stride=(2,2), padding=(1,1))
        
        self.conv2d4 = nn.Conv2d(256, 512, (3,3), (1,1), (1,1))
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d((3,3), stride=(2,2), padding=(1,1))
        
        self.conv2d5 = nn.Conv2d(512, 512, (3,3), (1,1), (1,1))
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d((3,3), stride=(2,2), padding=(1,1))
        
        self.conv2d6 = nn.Conv2d(512, 736, (1,1), stride=(1,1), padding=(0,0))
        
        self.out = nn.Linear(736, 6624)
        
        if useCuda:
            self.cuda(gpuDevice)



    def forward(self, x):
        

        #print("original shape: ", x.size())
        x = self.conv2d1(x)
        #print("shape after conv2d: ", x.size())
        x = self.relu1(x)
        #print("shape after first relu: ",x.size())
        x = self.pool1(x)
        #print("shape after maxpool2d: ", x.size())
        
        x = self.conv2d2(x)
        #print("shape after conv2d: ", x.size())
        x = self.relu2(x)
        #print("shape after first relu: ",x.size())
        x = self.pool2(x)
        #print("shape after maxpool2d: ", x.size())
       
        x = self.conv2d3(x)
        #print("shape after conv2d: ", x.size())
        x = self.relu3(x)
        #print("shape after first relu: ",x.size())
        x = self.pool3(x)
        #print("shape after maxpool2d: ", x.size())
        
        x = self.conv2d4(x)
        #print("shape after conv2d: ", x.size())
        x = self.relu4(x)
        #print("shape after first relu: ",x.size())
        x = self.pool4(x)
        #print("shape after maxpool2d: ", x.size())
        
        x = self.conv2d5(x)
        #print("shape after conv2d: ", x.size())
        x = self.relu5(x)
        #print("shape after first relu: ",x.size())
        x = self.pool5(x)
        print("shape after lastpool: ", x.size())
        
        output = self.conv2d6(x)
        print("final fc utput shape: ", output.size())
        
        return output

    @property
    def is_cuda(self):
       
        return next(self.parameters()).is_cuda

    def save(self, path):
        
        print('Saving model... %s' % path)
        torch.save(self, path)


