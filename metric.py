import torch.nn.functional as F
import torch
import numpy as np

class ValScore_depth(object):
    def __init__(self):
        self.a1 = 0
        self.a2 = 0
        self.a3 = 0
        self.rmse = 0
        self.rmse_log = 0
        self.abs_rel = 0
        self.sq_rel = 0
        self.silog = 0
        self.log_10 = 0
        self.count = 0
    
    def update(self, performances, N):
        self.count += N
        self.a1 += performances['a1'] * N
        self.a2 += performances['a2'] * N
        self.a3 += performances['a3'] * N
        self.rmse += performances['rmse'] * N
        self.rmse_log += performances['rmse_log'] * N
        self.abs_rel += performances['abs_rel'] * N
        self.sq_rel += performances['sq_rel'] * N
        self.silog += performances['silog'] * N
        self.log_10 += performances['log_10'] * N
    
    def average(self):
        self.a1 /= self.count
        self.a2 /= self.count
        self.a3 /= self.count
        self.rmse /= self.count
        self.rmse_log /= self.count
        self.abs_rel /= self.count
        self.sq_rel /= self.count
        self.silog /= self.count
        self.log_10 /= self.count

        return dict(a1=self.a1, a2=self.a2, a3=self.a3, abs_rel=self.abs_rel, rmse=self.rmse, log_10=self.log_10, rmse_log=self.rmse_log, silog=self.silog, sq_rel=self.sq_rel)


class ValScore_normal(object):
    def __init__(self):
        self.a11 = 0
        self.a22 = 0
        self.a30 = 0
        self.rmse = 0
        self.rmse_log = 0
        self.mean = 0
        self.median = 0
        self.count = 0
    
    def update(self, performances, N):
        self.count += N
        self.a11 += performances['a11'] * N
        self.a22 += performances['a22'] * N
        self.a30 += performances['a30'] * N
        self.rmse += performances['rmse'] * N
        self.mean += performances['mean'] * N
        self.median += performances['median'] * N
    
    def average(self):
        self.a11 /= self.count
        self.a22 /= self.count
        self.a30 /= self.count
        self.rmse /= self.count
        self.mean /= self.count
        self.median /= self.count

        return dict(a11=self.a11, a22=self.a22, a30=self.a30, rmse=self.rmse, mean=self.mean, median=self.median)


def compute_depth_errors(pred, gt, mask):
    median_scaling_factor = gt[mask > 0].median() / pred[mask > 0].median()
    pred *= median_scaling_factor
    
    thresh = torch.max((gt[mask>0] / pred[mask>0]), (pred[mask>0] / gt[mask>0]))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    abs_rel = torch.mean(torch.abs(gt[mask>0] - pred[mask>0]) / gt[mask>0])
    sq_rel = torch.mean(((gt[mask>0] - pred[mask>0]) ** 2) / gt[mask>0])

    rmse = (gt[mask>0] - pred[mask>0]) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt[mask>0]) - torch.log(abs(pred[mask>0]))) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    err = torch.log(pred[mask>0]) - torch.log(gt[mask>0])
    silog = torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100
    log_10 = (torch.abs(torch.log10(gt[mask>0]) - torch.log10(pred[mask>0]))).mean()

    N = mask.sum()

    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log, silog=silog, sq_rel=sq_rel), N



def compute_normal_errors(pred, gt, mask):
    mask = torch.tile(mask, (1, 3, 1, 1)) 
    mask = mask[:,0,:,:]
    mask = mask.type('torch.ByteTensor')

    error = angular_err(gt, pred)
    error = error[mask]

    a11 = (error < 11.25).float().mean()
    a22 = (error < 22.5).float().mean()
    a30 = (error < 30).float().mean()
    rmse = torch.sqrt((error ** 2).mean())
    mean = error.mean()
    median = error.median()

    N = mask.sum()

    return dict(a11=a11, a22=a22, a30=a30, rmse=rmse, mean=mean, median=median), N


def angular_err(gt, pred):
    prediction_error = torch.cosine_similarity(gt, pred, dim=1)
    prediction_error = torch.clamp(prediction_error, min=-1.0, max=1.0)
    err = torch.acos(prediction_error) * 180.0 / torch.pi
    return err