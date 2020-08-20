# ExpNet Phase1 Class Implementation
import torch
import torch.nn as nn
from utils.model_phase1_expnet_CPU import ExpNet


class ExpNet_p2(nn.Module):

    def __init__(self, useCuda=False, gpuDevice=0):
        super(ExpNet_p2, self).__init__()

        self.gpuDevice = gpuDevice

        self.expnet_p1 = ExpNet(useCuda, gpuDevice)
        self.expnet_p1.load_state_dict(torch.load('./model/expnet_p1.pt', \
                                                  map_location=lambda storage, loc: storage))

        self.phase1 = self.expnet_p1.eval()

        self.pool1 = nn.AvgPool2d((3, 3), stride=(2, 2), padding=(1, 1))
        self.pool2 = nn.AvgPool2d((3, 3), stride=(2, 2), padding=(1, 1))
        self.fc1 = nn.Linear(736, 256)
        self.fc2 = nn.Linear(256, 8)

    def forward(self, x):
        x = self.phase1(x)
        x = x.view((x.size(0), -1))
        x = self.fc1(x)
        output = self.fc2(x)

        return output

    def resize(self, data):
        size = data.size()
        if len(size) != 4:
            data = data.view(size[0], size[2], size[3], size[4])
        return data

    def save(self, path):
        print('Saving model... %s' % path)
        torch.save(self, path)
