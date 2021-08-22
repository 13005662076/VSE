import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

def l2norm(feature):
    y=torch.pow(feature,2).sum(dim=1,keepdim=True).sqrt()
    feature=torch.div(feature,y)
    return feature

class ImageModel(nn.Module):
    def __init__(self,model,out_size):
        super(ImageModel,self).__init__()
        self.out_size=out_size
        self.feature=model.features
        self.classifier=model.classifier
        
        #冻结特征提取层的参数，不参与训练
        for parameter in self.feature.parameters():
            parameter.requires_grad=False

        #改变网络最后一层的输出
        self.classifier[6].out_features=out_size
        
        

    def forward(self,x):
        x=self.feature(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x





#print(ImageModel(models.vgg16(pretrained=False),1024))
#print(models.vgg16(pretrained=False))#.classifier[6].in_features
#print(create_model(8))
