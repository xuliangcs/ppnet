import torch
import torch.nn as nn
import torch.nn.functional as F


class ppnet(torch.nn.Module):
    def __init__(self, num_classes):
        super(ppnet, self).__init__()    

        self.layer1 = torch.nn.Sequential()
        self.layer1.add_module("conv", torch.nn.Conv2d(1, 16, 5, 1))#5#3#7
        self.layer1.add_module("bn", torch.nn.BatchNorm2d(16))

        self.layer2 = torch.nn.Sequential()
        self.layer2.add_module("conv", torch.nn.Conv2d(16, 32, 1, 1))
        self.layer2.add_module("bn", torch.nn.BatchNorm2d(32, momentum=0.001, affine=True, track_running_stats=True))
        self.layer2.add_module("sigmoid", torch.nn.Sigmoid())
        self.layer2.add_module("avgpool", torch.nn.AvgPool2d(2, 2))
        
        self.layer3 = torch.nn.Sequential()
        self.layer3.add_module("conv", torch.nn.Conv2d(32, 64, 3, 1))
        self.layer3.add_module("bn", torch.nn.BatchNorm2d(64, momentum=0.001, affine=True, track_running_stats=True))
        self.layer3.add_module("sigmoid", torch.nn.Sigmoid())
        self.layer3.add_module("avgpool", torch.nn.AvgPool2d(2, 2))
        
        self.layer4 = torch.nn.Sequential()
        self.layer4.add_module("conv", torch.nn.Conv2d(64, 64, 3, 1))
        self.layer4.add_module("bn", torch.nn.BatchNorm2d(64, momentum=0.001, affine=True, track_running_stats=True))
        self.layer4.add_module("relu", torch.nn.ReLU())


        self.layer5 = torch.nn.Sequential()
        self.layer5.add_module("conv", torch.nn.Conv2d(64, 256, 3, 1))
        self.layer5.add_module("bn", torch.nn.BatchNorm2d(256, momentum=0.001, affine=True, track_running_stats=True))
        self.layer5.add_module("relu", torch.nn.ReLU())
        self.layer5.add_module("maxpool", torch.nn.MaxPool2d(2,2))


        self.fc1 = torch.nn.Linear(43264, 512)
        self.bn1 = torch.nn.BatchNorm1d(512, momentum=0.001, affine=True, track_running_stats=True)
        self.relu1 = torch.nn.ReLU()

        
        self.fc2 = torch.nn.Linear(512, 512)
        self.bn2 = torch.nn.BatchNorm1d(512, momentum=0.001, affine=True, track_running_stats=True)
        self.relu2 = torch.nn.ReLU()
        self.drop2 = torch.nn.Dropout(p=0.25)


        self.dis = torch.nn.PairwiseDistance(p=2,)   


        self.fc3 = torch.nn.Linear(512, num_classes)
        

        # self.featuremap = torch.nn.Sequential()
        # self.featuremap.add_module('layer1', self.layer1)
        # self.featuremap.add_module('layer2', self.layer2)
        # self.featuremap.add_module('layer3', self.layer3)
        # self.featuremap.add_module('layer4', self.layer4)
        # self.featuremap.add_module('layer5', self.layer5)

        # self.code = torch.nn.Sequential()
        # self.code.add_module('fc1', self.fc1)
        # self.code.add_module('bn1', self.bn1)
        # self.code.add_module('relu1', self.relu1)
        # self.code.add_module('fc2', self.fc2)
        # self.code.add_module('bn2', self.bn2)
        # self.code.add_module('relu2', self.relu2)
        # self.code.add_module('drop2', self.drop2)

        # self.classifier = torch.nn.Sequential()
        # self.classifier.add_module('fc3', self.fc3)




    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = x.view(x.size(0), -1) 
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)


        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)


        b, _ = x.size()
        o1 = x[:b//2, :]
        o2 = x[b//2:, :]
        dis = self.dis(o1, o2)      


        x = self.fc3(x)

        return x, dis



    def getFeatureCode(self, x):   
        
        # x = self.featuremap(x)
        # x = x.view(x.size(0), -1)         
        # x = self.code(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = x.view(x.size(0), -1) 
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)


        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
       
        return x


