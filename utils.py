import torch.nn.functional as F
import torch
import numpy as np
from math import pi


def compute_normals_full_accelerate(depth):
    normals = torch.zeros((depth.shape[0], 3, depth.shape[-2], depth.shape[-1]), dtype=torch.float32).cuda()
    angles = torch.zeros((depth.shape[0], 2, depth.shape[-2], depth.shape[-1]), dtype=torch.float32).cuda()
    points = torch.zeros((depth.shape[0], 3, depth.shape[-2], depth.shape[-1]), dtype=torch.float32).cuda()

    theta = torch.linspace(-pi, pi, depth.shape[-1]).cuda()
    phi = torch.linspace(-pi/2, pi/2, depth.shape[-2]).cuda()
    theta = torch.tile(theta, (depth.shape[-2], 1))
    phi = torch.tile(phi, (depth.shape[-1], 1)).T

    angles[:, 0, :, :] = theta
    angles[:, 1, :, :] = phi
    
    points[:, 0, :, :] = torch.cos(angles[:,1,:,:]) * torch.sin(angles[:,0,:,:]) * depth.squeeze(1)   # X
    points[:, 1, :, :] = torch.cos(angles[:,1,:,:]) * torch.cos(angles[:,0,:,:]) * depth.squeeze(1)   # Y
    points[:, 2, :, :] = torch.sin(angles[:,1,:,:]) * depth.squeeze(1)                                # Z

    # normalized normal map
    vec0_pad = F.pad(points[:, :, :, :-1] - points[:, :, :, 1:], pad=[0, 1, 0, 0, 0, 0, 0, 0], mode='constant', value=0) 
    vec2_pad = F.pad(points[:, :, :-1, :] - points[:, :, 1:, :], pad=[0, 0, 0, 1, 0, 0, 0, 0], mode='constant', value=0)
    vec4_pad = F.pad(points[:, :, :, 1:] - points[:, :, :, :-1], pad=[1, 0, 0, 0, 0, 0, 0, 0], mode='constant', value=0)
    vec6_pad = F.pad(points[:, :, 1:, :] - points[:, :, :-1, :], pad=[0, 0, 1, 0, 0, 0, 0, 0], mode='constant', value=0)

    cross20 = torch.cross(vec2_pad, vec0_pad)
    cross42 = torch.cross(vec4_pad, vec2_pad)
    cross64 = torch.cross(vec6_pad, vec4_pad)
    cross06 = torch.cross(vec0_pad, vec6_pad)

    normalmap = F.normalize(cross20)
    normalmap += F.normalize(cross42)
    normalmap += F.normalize(cross64)
    normalmap += F.normalize(cross06)
    normalmap = F.normalize(normalmap)
    
    mask = (normalmap == 0).all(dim=1).cuda()
    expanded_mask = mask.unsqueeze(1).expand_as(normalmap)
    normalmap[expanded_mask] = 1/3**(0.5)
    
    return normalmap