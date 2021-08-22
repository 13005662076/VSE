import torch
import torch.nn as nn


#定义ConTrastiveLoss
class ContrastiveLoss(nn.Module):
    #初始化margin参数
    def __init__(self,margin=0,measure=False,max_violation=False):
        super(ContrastiveLoss,self).__init__()
        self.margin=margin
        self.max_violation=max_violation
    
    def forward(self,im,s):
        #计算图像-文本的分数矩阵
        scores=im.mm(s.t())
        diagonal=scores.diag().view(im.size(0),1)
        d1=diagonal.expand_as(scores)
        d2=diagonal.t().expand_as(scores)
        #正样本和负样本矩阵
        #比较输出矩阵中对角线上的分数与每列元素（文本）的分数
        cost_s=(self.margin+scores-d1).clamp(min=0)
        #比较输出矩阵中对角线上的分数与每行元素（图像）的分数
        cost_im=(self.margin+scores-d2).clamp(min=0)
        
        mask=torch.eye(scores.size(0))>0.5
        if torch.cuda.is_available():
            mask=mask.cuda()
        cost_s=cost_s.masked_fill_(mask,0)
        cost_im=cost_im.masked_fill_(mask,0)
        #记录最大负样本，并计算norm
        if self.max_violation:
            cost_s=cost_s.max(1)[0]
            cost_im=cost_im.max(0)[0]
        return cost_s.sum()+cost_im.sum()

    
#定义NCE LOSS
class NceLoss(nn.Module):
    #初始化batch size以及top k hard sample参数
    def __init__(self,top_k=2,scale=100.0):
        super(TopK_MulCLSLoss,self).__init__()
        self.top_k=top_k
        self.scale=scale
        
        
    def forward(self,im,s):
        #计算图像-文本的分数矩阵
        scores=im.mm(s.t())
        diag=scores.diag()
        mask=torch.eye(scores.size(0))>0.5
        if torch.cuda.is_available():
            mask=mask.cuda()

        targets=torch.LongTensor(np.zeros(im.size(0)))#.cuda()
        
        scores=scores.masked_fill_(mask,3,0)
        #分数由高到低排序，并取topK结果
        s_i2t,_=torch.sort(scores,dim=1,descending=True)
        s_i2t,_=s_i2t[:,:self.top_k]
        s_i2t[:,0]=diag
        s_t2i,_=torch.sort(scores.t(),dim=1,descending=True)
        s_t2i=s_t2i[:,:self.top_k]
        s_t2i=diag
        s_i2t=self.scale*s_i2t
        s_t2i=self.scale*s_t2i
        return nn.CrossEntropyLoss()(s_i2t,targets)+self.mulcls(s_t2i,targets)
