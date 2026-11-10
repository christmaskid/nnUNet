## CLDice Loss implementation: [CVPR 2021]
## Ref.: https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/cldice.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss

# Ref.: https://arxiv.org/pdf/2003.07311
# Original paper uses k=5~25 (num_iter)
class SoftSkeletonize(torch.nn.Module):

    def __init__(self, num_iter=5):
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img): # minpool(I)
        # minpool is not implemented in torch.functional, but can be implemented as maxpool(-I)
        img = img.float()

        if len(img.shape)==4:
            p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
            p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
            return torch.min(p1,p2)
        elif len(img.shape)==5:
            p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
            p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
            p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img): # maxpool(I)
        img = img.float()

        if len(img.shape)==4:
            return F.max_pool2d(img, (3,3), (1,1), (1,1))
        elif len(img.shape)==5:
            return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))

    def soft_open(self, img): # maxpool(minpool(I))
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        img1 = self.soft_open(img) # I' = maxpool(minpool(I))
        skel = F.relu(img-img1) # S = ReLU(I-I')

        for j in range(self.num_iter):
            img = self.soft_erode(img) # I  = minpool(I)
            img1 = self.soft_open(img) # I' = maxpool(minpool(I))
            skel = skel + (1 - skel) * F.relu(img - img1) # S = S + (1-S)*ReLU(I-I')

        return skel

    def forward(self, img):
        img = img.float()

        return self.soft_skel(img)


class soft_cldice(nn.Module):
    def __init__(self, iter_=5, smooth = 1e-6, exclude_background=False):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=self.iter)
        self.exclude_background = exclude_background

    def forward(self, y_true, y_pred):
        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice = 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice

class SoftCLDiceLoss(nn.Module):
    def __init__(self, iter_=5, smooth = 1e-6, exclude_background=False):
        super(SoftCLDiceLoss, self).__init__()
        self.cldice = soft_cldice(iter_=iter_, smooth=smooth, exclude_background=exclude_background)

    def forward(self, y_true, y_pred):
        return 1-self.cldice(y_true, y_pred)

class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=5, alpha=0.5, smooth = 1e-6, exclude_background=False):
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.exclude_background = exclude_background

        self.soft_dice = MemoryEfficientSoftDiceLoss(
            apply_nonlin=torch.sigmoid, 
            do_bg=not exclude_background, 
            smooth=smooth)
        self.soft_skeletonize = SoftSkeletonize(num_iter=self.iter)

    def forward(self, y_true, y_pred):
        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
        dice = self.soft_dice(y_true, y_pred) # dice = soft_dice(y_true, y_pred)
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return (1.0-self.alpha)*dice+self.alpha*cl_dice

class SoftDiceCLDiceLoss(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth = 1., exclude_background=False):
        super(SoftDiceCLDiceLoss, self).__init__()
        self.cldice = soft_dice_cldice(iter_=iter_, alpha=alpha, smooth=smooth, exclude_background=exclude_background)

    def forward(self, y_true, y_pred):
        return 1-self.cldice(y_true, y_pred)
