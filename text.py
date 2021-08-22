import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import numpy as np

def l2norm(feature):
    y=torch.pow(feature,2).sum(dim=1,keepdim=True).sqrt()
    feature=torch.div(feature,y)
    return feature

class Text(nn.Module):
    def __init__(self,n_word,word_dim,out_size,num_layers=3):
        super(Text,self).__init__()
        self.out_size=out_size
        #构造词向量
        self.embedding=nn.Embedding(n_word,word_dim)

        self.rnn=nn.LSTM(word_dim,out_size,num_layers,batch_first=True)
        
        #初始化网络权重
        self.init_weights()
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1,0.1)
        
    #定义网络前向：文本特征提取过程
    def forward(self,x,length):
        #通过查询word ids,获取每个word的feature,并进行embedding
        x=self.embedding(x)
        pack_x=pack_padded_sequence(x,length,batch_first=True)
        #将embedding的每个word feature 灌入RNN
        out,_=self.rnn(pack_x)
        #Reshape输出向量的大小
        pad_x=pad_packed_sequence(out,batch_first=True)

        I=torch.LongTensor(length).view(-1,1,1)
        I=Variable(I.expand(x.size(0),1,self.out_size)-1)

        out=torch.gather(pad_x[0],1,I).squeeze(1)
        #将模型输出的vector归一化
        out=l2norm(out)
        return out
        
        
