import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch


class BerhuLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerhuLoss, self).__init__()
        self.threshold = threshold

    def forward(self, pred, target, mask=None, interpolate=True):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        if interpolate:
            pred = nn.functional.interpolate(pred, target.shape[-2:], mode='bilinear', align_corners=True)
        valid_mask = (target > 0).detach()
        if mask is not None:
            valid_mask *= mask.detach()

        diff = torch.abs(target - pred)
        diff = diff[valid_mask]
        delta = self.threshold * torch.max(diff).data.cpu().numpy()

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff ** 2 + delta ** 2, 2.0 * delta ** 2, 0.)
        part2 = part2 / (2. * delta)
        diff = part1 + part2
        loss = diff.mean()
        return loss


class Cossim(nn.Module):
    def __init__(self):
        super(Cossim, self).__init__()
    
    def forward(self, pred, gt, mask, interpolate=True):
        if interpolate:
            pred = nn.functional.interpolate(pred, gt.shape[-2:], mode='bilinear', align_corners=True)
        pred *= mask
        gt *= mask

        similarity = 1 - F.cosine_similarity(pred, gt, dim=1)
        valid_pixel = torch.sum(mask, dim=[1, 2, 3], keepdim=True).float()
        masked_err = similarity * mask.squeeze(1).float()
        return torch.mean(torch.sum(masked_err, dim=[-2, -1], keepdim=True) / valid_pixel)    


class NormalL1Loss(nn.Module):
    def __init__(self):
        super(NormalL1Loss, self).__init__()
    
    def forward(self, pred, gt, mask, interpolate=True):
        if interpolate:
            pred = nn.functional.interpolate(pred, gt.shape[-2:], mode='bilinear', align_corners=True)

        l1 = torch.sum(torch.abs(gt - pred), dim=1, keepdim=True)
        loss = torch.mean(l1[mask])
        return loss


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
        self.x_kernel = [[0, -1, 0],[0, 0, 0], [0, 1, 0]]
        self.y_kernel = [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]
        
        self.x_kernel = torch.FloatTensor(self.x_kernel).unsqueeze(0).unsqueeze(0)
        self.y_kernel = torch.FloatTensor(self.y_kernel).unsqueeze(0).unsqueeze(0)
        self.weight_x = nn.Parameter(data=self.x_kernel, requires_grad=False).cuda()
        self.weight_y = nn.Parameter(data=self.y_kernel, requires_grad=False).cuda()
    
    def forward(self, pred, target, mask=None, interpolate=True):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        if interpolate:
            pred = nn.functional.interpolate(pred, target.shape[-2:], mode='bilinear', align_corners=True)
        valid_mask = (target > 0).detach()
        if mask is not None:
            valid_mask *= mask.detach()

        gx = F.conv2d((pred - target), self.weight_x, padding=1)
        gy = F.conv2d((pred - target), self.weight_y, padding=1)
        loss = (abs(gx) + abs(gy)).mean()
        return loss