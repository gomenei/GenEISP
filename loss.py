import torch
import torch.nn as nn
import torch.nn.functional as F

# for example
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, predictions, targets):
        return F.mse_loss(predictions, targets)

class L_TV_L1(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV_L1,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        # batch_size = x.size()[0]
        h_x = x.size()[0]
        w_x = x.size()[1]
        count_h =  (x.size()[0]-1) * x.size()[1]
        count_w = x.size()[0] * (x.size()[1] - 1)
        h_tv = torch.abs((x[1:,:]-x[:h_x-1,:])).sum()
        w_tv = torch.abs((x[:,1:]-x[:,:w_x-1])).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)
    
class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        # batch_size = x.size()[0]
        h_x = x.size()[0]
        w_x = x.size()[1]
        count_h =  (x.size()[0]-1) * x.size()[1]
        count_w = x.size()[0] * (x.size()[1] - 1)
        h_tv = torch.pow((x[1:,:]-x[:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,1:]-x[:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)