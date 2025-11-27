import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import datetime
import random
import torch
import os

from torch.utils.data import DataLoader, Dataset
from Dataloaders.dataset_loader_stanford import Dataset
# from Dataloaders.dataset_loader_matterport import Dataset
# from Dataloaders.dataset_loader_structured3d import Dataset

from loss import BerhuLoss, Cossim, GradLoss, NormalL1Loss
from models.hush import HUSH
from tqdm import tqdm
from metric import *
from utils import *


def main(args):
    input_dir = args.rootpath
    train_file = args.trainfile
    val_file = args.valfile
    batch_size = args.bs
    epochs = args.epoch
    
    train_loader = torch.utils.data.DataLoader(dataset = Dataset(rotate = True, flip = True, root_path=input_dir, path_to_img_list=train_file),
                                                batch_size=batch_size, shuffle=True, num_workers=16, drop_last = True, pin_memory=True )
    val_loader = torch.utils.data.DataLoader(dataset = Dataset(rotate = False, flip = False, root_path=input_dir, path_to_img_list=val_file),
                                                batch_size=1, shuffle=False, num_workers=16, drop_last = True, pin_memory=True)


    Net = HUSH(min_val=1e-3, max_val=10, degree=10).cuda()
    optimizer = optim.AdamW(list(Net.parameters()), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6, last_epoch=-1)
    
    criterion_depth = BerhuLoss()
    criterion_normal = Cossim()
    criterion_grad = GradLoss()
    best_abs = np.inf
    best_rmse = np.inf

    for epoch in range(epochs):
        Net.train()
        for batch, (rgb, depth, depth_mask) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
        
            rgb, depth, depth_mask  = rgb.cuda(), depth.cuda(), depth_mask.cuda()
            normal = compute_normals_full_accelerate(depth)
            pred_depth, pred_normal = Net(rgb)

            depth_loss = criterion_depth(pred_depth, depth, mask=depth_mask.to(torch.bool))
            normal_loss = criterion_normal(pred_normal, normal, mask=depth_mask.to(torch.bool))
            grad_loss = criterion_grad(pred_depth, depth, mask=depth_mask.to(torch.bool))
            loss = depth_loss + normal_loss + 0.5*grad_loss

            if batch % 20 == 0 and batch > 0:
                print('[Epoch %d--Iter %d]Total loss %.4f' % (epoch, batch, loss))
                
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        ## Validation per 1 epoch 
        Net.eval()
        val_performance = validate_HUSH(Net, val_loader)
        
        if val_performance['rmse'] < best_rmse:
            torch.save(Net.state_dict(), './best_model_SF2D3D.pth')
            best_rmse = val_performance['rmse']


def validate_HUSH(Net, val_loader):
    val_metrics = ValScore_depth()
    with torch.no_grad():
        for batch, (rgb, depth, depth_mask) in tqdm(enumerate(val_loader)):
            rgb, depth, depth_mask = rgb.cuda(), depth.cuda(), depth_mask.cuda()
            pred_depth, _ = Net(rgb)

            pred_depth = F.interpolate(pred_depth, depth.shape[-2:], mode='bilinear', align_corners=True)
            performance, N = compute_depth_errors(pred_depth, depth, depth_mask)
            val_metrics.update(performance, N)
        performances = val_metrics.average()
        print("Validation score :", performances)
    
    return performances


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootpath', type=str, default='/home/SF2D3D/')
    parser.add_argument('--trainfile', type=str, default='./filenames/train_stanford2d3d.txt')
    parser.add_argument('--valfile', type=str, default='./filenames/test_stanford2d3d.txt')

    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('-g', '--gpu_id', default=0, type=int, help='gpu id setting')
    args = parser.parse_args()
    
    # GPU ID setting 
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    # Fix seed 
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    main(args)