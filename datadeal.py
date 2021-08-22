import torch
import os,cv2
import nltk
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from random import shuffle
from vocab import *

#数据集处理模块
class DatasetP(Dataset):
    def __init__(self,filename,transform=None):
        
        image_path,self.cap=[],[]
        self.lengths=[]
        n=0
        self.images=[]
        root="E:/testdl/VSE/Flickr8k and Flickr8kCN/Flicker8k_Dataset/"
        with open(filename, 'r+') as f:
            for line in f:
                n+=1
                img,ca=line.split("#enc#")[0].strip(),line.split("#enc#")[1][2:].strip()

                filepath=os.path.join(root,img)
                data=cv2.imread(filepath,1)
                
                data = cv2.resize(data, (224, 224))
        
                self.images.append(data)
                
                capn=nltk.tokenize.word_tokenize(str(ca).lower())[:-1]
                self.cap.append(capn)
                self.lengths.append(len(capn))
                if n>200:
                    break
        
        self.transform=transform
        self.dat=[]
        self.max=max(self.lengths)
        self.v=create_dict(self.cap)

        for i,j,z in  zip(self.images,self.cap,self.lengths):
            self.dat.append([i,j,z])
    
        self.sshuffle()

    def __getitem__(self, index):
        
        img = self.images[index]
        img = img.transpose((2,0,1))
        if self.transform is not None:
            img = self.transform(img)
        caption = []
        caption.extend([self.v(token) for token in self.cap[index]])

        for i in range(self.max-len(caption)):
            caption.append(0)
        cap = torch.LongTensor(caption)
        
        return img/255, cap,self.lengths[index], index
        
    def __len__(self):
        return len(self.images)
        

    #对数据集进行打乱
    def sshuffle(self):
        shuffle(self.dat)
        self.dat=np.array(self.dat)
        self.images.clear()
        self.cap.clear()
        self.lengths.clear()
        for i,j,z in zip(self.dat[:,0],self.dat[:,1],self.dat[:,2]):
            self.images.append(i)
            self.cap.append(j)
            self.lengths.append(z)
    def get_cap(self):
        return np.array(self.cap)
    def get_images(self):
        return np.array(self.images)
#d=DatasetP('E:/testdl/VSE/Flickr8k and Flickr8kCN/flickr8kcn/data/train.txt')   
#print(d.__len__())
