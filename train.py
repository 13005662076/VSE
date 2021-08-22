from torchvision import transforms
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
import os
import numpy as np
from datetime import datetime
from datadeal import *
from image import *
from text import *
from loss import *

def create_model(out_size=1000):
    image_model=models.vgg16(pretrained=True)
    text_model=Text(40455,300,out_size)
    return ImageModel(image_model,out_size),text_model

def start_train(image_model,text_model,im,cap,length):
    img_output=image_model(im)
    text_output=text_model(cap,length)
    return img_output,text_output

def Vse():
    '''
    tf=transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    '''
    #train_set,train_labels=get_valid("E:/testdl/train")

    #获得训练数据
    train_set=DatasetP("E:/testdl/VSE/Flickr8k and Flickr8kCN/flickr8kcn/data/train.txt")
    train_loader=DataLoader(train_set,batch_size=32,shuffle=False,num_workers=0)

    #获得验证数据
    test_set=DatasetP("E:/testdl/VSE/Flickr8k and Flickr8kCN/flickr8kcn/data/test.txt")
    test_loader=DataLoader(test_set,batch_size=16,shuffle=False,num_workers=0)

    image_model,text_model=create_model()
    print(image_model)
    params = list(image_model.classifier.parameters())
    params += list(text_model.parameters())
    
    optimizer=optim.Adam(params,lr=0.001)
    
    criterion = ContrastiveLoss()
    #criterion=NceLoss()
    for epoch in range(150):
        train_loss=0.0
        for im,cap,length,batch_index in train_loader:
            length,index=torch.sort(length,descending=True)
            cap=torch.index_select(cap,0,index)
            im=im.type('torch.FloatTensor')
            if torch.cuda.is_available():
                im,cap=Variable(im.cuda()),Variable(cap.cuda())
                
            else:
                im,cap=Variable(im),Variable(cap)
                
                
            #model.zero_grad()
            output1,output2=start_train(image_model,text_model,im,cap,length)
            
            loss=criterion(output1,output2)
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()

        
        #valid_test  使用验证集来测试模型
        val_loss=0.0
        for im,cap,length,batch_index in test_loader:
            length,index=torch.sort(length,descending=True)
            cap=torch.index_select(cap,0,index)
            im=im.type("torch.FloatTensor")
            if torch.cuda.is_available():
                im,cap=Variable(im.cuda()),Variable(cap.cuda())
            else:
                im,cap=Variable(im),Variable(cap)
            output1,output2=start_train(image_model,text_model,im,cap,length)
            
            loss=criterion(output1,output2)
            val_loss+=loss.item()
        
        
        print("Epoch %d. train_loss:%f val_loss:%f"%(epoch+1,train_loss/len(train_loader),train_loss/len(test_loader)))
            
Vse()



