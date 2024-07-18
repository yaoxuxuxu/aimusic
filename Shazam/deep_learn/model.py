import torch
from torch import nn
import numpy as np
from torchvision import models


class mynet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        self.relu=nn.ReLU(True)
        self.bn=nn.BatchNorm2d(3)
        self.resnet=models.resnet50()
        self.lin=nn.Linear(2048,620)
        #print(self.resnet)

    def forward(self,x):
        x = self.relu(self.bn(self.conv1(x)))
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = self.lin(x)
        """x = x.view(x.size(0), -1)
        x=self.test(x)"""
        return x

model=mynet()
shape=np.array([1,1,130,130])
x=np.random.random(shape)
print(x.shape)
x=torch.tensor(x,dtype=torch.float32)
model(x)
