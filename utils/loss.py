import torch
#torch.set_printoptions(profile="full")
import torch.nn as nn
import torch.nn.functional as F
class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'dict':
            return self.Dictloss
        elif mode =='total':
            return self.Total
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()

        #有softmax()的步骤
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.25):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss


    def BinaryDiceLoss(self,logit, targets):
        # 获取每个批次的大小 N
        N = logit.size()[0]

        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        logit = logit.view(N, -1)

        #print(input_flat)
        targets = targets.view(N, -1)
        #print(targets_flat)

        # 计算交集
        #print(logit.shape,targets.shape)
        intersection = logit * targets
        #print(intersection)
        dice_eff = (2 * intersection.sum(1) + smooth) / (logit.sum(1) + targets.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        loss = 1 - dice_eff.sum() / N
        return loss

    def Dictloss(self,logit,target):

        nclass = logit.shape[1]
        #print(nclass)
        target = torch.nn.functional.one_hot(target.long(),nclass)
        target = target.transpose(1,3)
        target = target.transpose(2,3)
        #print(logit.shape)
        #print(target.shape)

        assert logit.shape == target.shape, "predict & target shape do not match"

        total_loss = 0
        logit1 = F.softmax(logit,dim=1)
        c = target.shape[1]

        for i in range(c):
            dict_loss = self.BinaryDiceLoss(logit1[:,i], target[:,i])
            total_loss += dict_loss

        return total_loss / c

   
if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(2, 3, 7,7).cuda()
    b = torch.rand(2, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
    print(loss.Dictloss(a,b).item())




