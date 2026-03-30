#week 8 - neural networks - RNNs

#from last week's material
#saved a file named nnhelpers.py

# this week's material:

#for reading data
from torch.utils.data import DataLoader
#for visualizing
import plotly.express as px
#for model building
import torch
import torch.nn as nn
import torch.nn.functional as F
#import helpers
import nnhelpers as nnh

#still loading the same data as last week!
#load our data into memory
train_data = nnh.CustomMNIST("https://github.com/dustywhite7/Econ8310/raw/master/DataSets/mnistTrain.csv")
test_data = nnh.CustomMNIST("https://github.com/dustywhite7/Econ8310/raw/master/DataSets/mnistTest.csv")

#create data feed pipelines for modeling
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        #6 output channels, 5x5 square convolution
        #kernel
        self.conv1 = nn.LazyConv2d(6, 5, padding=2)                   #convolution/transformation 6 layers, 5x5, padding with 2
        self.conv2 = nn.LazyConv2d(16, 5)
        #an affine operation: y = Wx + b
        self.fc1 = nn.LazyLinear(120)                              #fully connected flat layer. example 120 neurons
        self.fc2 = nn.LazyLinear(84)
        self.fc3 = nn.LazyLinear(10)                             #10 perceptrons. 1 per class in order to do classification

    def forward(self, x):
        #max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        #if the size is square, you can specify with a single number           #feed forward structure of netork
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)                         #pooling functions. reducing resolution
        #flatten all dimensions except the batch dimension
        x = torch.flatten(x, 1)                                         #flatten, fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))                                       #activation layers between connected layers
        x = self.fc3(x)
        return x                                                    #output. can be data passed on to next layer/block of network
                                                                            # or output of overall model

#create a model instance, pass the model to GPU
# model = LeNet()      #faster if add ".to('cuda')"
#then we train
# model = nnh.train_net(model, train_dataloader, test_dataloader,
#     epochs=5, learning_rate=1e-3, batch_size=64)



# NEW MODEL!!! ResNet-18
class Inception(nn.Module):
    # c1--c4 are the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
        # Branch 2
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
        # Branch 3
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        # Branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, x):
        b1 = F.relu(self.b1_1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_2(self.b4_1(x)))
        return torch.cat((b1, b2, b3, b4), dim=1)                        #concatenating all results from other pathways. joining
                                                                              #into single array so can pass it off to next layer

class Residual(nn.Module):
    """The Residual block of ResNet models."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):          #86 to 107 is just 1 block of residual network
        super(Residual, self).__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1,
                                   stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()                         #normalize. rescales and centers data to keep from becoming infinite
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))                  #activation
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X                                                #concatenating x and y
        return F.relu(Y)                                    #activation

class ResNet(nn.Module):                                         #describing architecture. any size resnet
    def __init__(self, arch, lr=0.1, num_classes=10):
        super(ResNet, self).__init__()
        self.net = nn.Sequential(self.b1())                     #creating block 1. specialized block which why separate def
        for i, b in enumerate(arch):
            self.net.add_module(f'b{i+2}', 
                self.block(*b, first_block=(i==0)))                #for loop repetedly created blocks from definition below
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),              #pooling, flattening, creating fully connected layer
            nn.LazyLinear(num_classes)))

    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def block(self, num_residuals, num_channels, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(num_channels,
                 use_1x1conv=True, strides=2))               #adding 1x1 convolution to first block in each group
            else:
                blk.append(Residual(num_channels))
        return nn.Sequential(*blk)

    def forward(self, x):
        x = self.net(x)                             #saying 'use the network we just described above'
        return x

class ResNet18(ResNet):
    def __init__(self, lr=0.1, num_classes=10):
        super(ResNet18, self).__init__(((2, 64), (2, 128),          #describes the block structure. ex: 2 block with 64 channels
         (2, 256), (2, 512)),
                       lr, num_classes)                   #passes in learning rate and number of classes we want to train model on

model = ResNet18()#.to('cuda')

model = nnh.train_net(model, train_dataloader, 
        test_dataloader, epochs = 5, learning_rate = 1e-3,
        batch_size=64
        )