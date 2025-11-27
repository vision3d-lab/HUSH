import torch.nn as nn
import torch
import importlib
import cv2

from einops import rearrange
from functools import partial
from timm.models.vision_transformer import _cfg
from .backbones.swin_encoder import SwinTransformer
from .backbones.defattn_decoder import MSDeformAttnPixelDecoder
from .backbones.attention import AttentionLayer
from .utils import (PositionEmbeddingSine, _get_activation_cls, get_norm)


def swin_large_22k(pretrained: bool = True, **kwargs):
    if pretrained:
        pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth"
    model = SwinTransformer(
        patch_size=4,
        window_size=7,
        embed_dims=[192 * (2**i) for i in range(4)],
        num_heads=[6, 12, 24, 48],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 18, 2],
        drop_path_rate=0.2,
        pretrained=pretrained,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


def MSDA_PixelDecoder():
    model = MSDeformAttnPixelDecoder.build()
    return model


class SH_Coeff_Extractor(nn.Module):
    def __init__(self):
        super(SH_Coeff_Extractor, self).__init__()
        self.dense1 = nn.Conv2d(256, 192, kernel_size=5, stride=2, bias=False)
        self.dense2 = nn.Conv2d(192, 128, kernel_size=5, stride=2, bias=False)
        self.dense3 = nn.Conv2d(128, 64, kernel_size=5, stride=2, bias=False)
        self.final_dense = nn.Linear(64*13*29, 55)
        
        self.bn1 = nn.BatchNorm2d(192)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.dense1(x)))
        x = self.relu(self.bn2(self.dense2(x)))
        x = self.relu(self.bn3(self.dense3(x)))
        
        x = torch.flatten(x, 1)
        coeffs = self.final_dense(x)
        return coeffs


class SH_Attention_Module(nn.Module):
    def __init__(self):
        super(SH_Attention_Module, self).__init__()
        self.CA = AttentionLayer(sink_dim=128,
                                hidden_dim=128,
                                source_dim=128,
                                output_dim=128,
                                num_heads=1,
                                dropout=0.0,
                                pre_norm=True,
                                sink_competition=True)

        self.MLP_depth = nn.Sequential(get_norm("torchLN", 128),
                                        nn.Linear(128, 4 * 128),
                                        _get_activation_cls("gelu"),
                                        nn.Linear(4 * 128, 128),)

        self.proj_output_depth = nn.Sequential(get_norm("torchLN", 128),
                                        nn.Linear(128, 128),
                                        get_norm("torchLN", 128),
                                        nn.Linear(128, 1),)

        self.MLP_normal = nn.Sequential(get_norm("torchLN", 128),
                                        nn.Linear(128, 4 * 128),
                                        _get_activation_cls("gelu"),
                                        nn.Linear(4 * 128, 128),)

        self.proj_output_normal = nn.Sequential(get_norm("torchLN", 128),
                                        nn.Linear(128, 128),
                                        get_norm("torchLN", 128),
                                        nn.Linear(128, 3),)

        self.SH_patch_emb = nn.Conv2d(55, 55, kernel_size=32, stride=32, padding=0)
        self.F_patch_emb0 = nn.Conv2d(256, 128, kernel_size=4, stride=4, padding=0)
        self.F_patch_emb1 = nn.Conv2d(256, 128, kernel_size=8, stride=8, padding=0)
        self.F_patch_emb2 = nn.Conv2d(256, 128, kernel_size=16, stride=16, padding=0)

        self.conv_1x1_depth = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding='same', bias=False)
        self.conv_1x1_normal = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding='same', bias=False)
        self.PE = nn.Parameter(torch.zeros(1, 128, 128)).cuda()

    def cal_sim(self, SH, erp_feature):
        SH_normed = torch.norm(SH, p=2, dim=-1).unsqueeze(-1).unsqueeze(-1)

        erp_feature_normed = torch.norm(erp_feature, p=2, dim=1).unsqueeze(1)
        similarity_map = torch.einsum('bne, behw -> bnhw', SH, erp_feature)
        similarity_map = similarity_map / SH_normed
        similarity_map = similarity_map / erp_feature_normed
        similarity_max_map, similarity_index_map = torch.max(similarity_map, dim=1, keepdim=True)

        one_hot = torch.FloatTensor(similarity_index_map.shape[0], SH.shape[1],
                                    similarity_index_map.shape[2], similarity_index_map.shape[3]).zero_().to(SH.device)
        similarity_index_map = one_hot.scatter_(1, similarity_index_map, 1)
        return similarity_index_map

    def Cross_Attn(self, Q, KV):
        return self.CA(Q, KV) + Q

    def Depth_head(self, feature, SH):
        _, _, H, _ = feature.shape
        feature_flat = rearrange(feature.clone(), "b c h w -> b (h w) c")

        update = self.CA(feature_flat, SH)
        feature_flat = feature_flat + update
        feature_flat = feature_flat + self.MLP_depth(feature_flat.clone())
        output = self.proj_output_depth(feature_flat)
        output = rearrange(output, "b (h w) c -> b c h w", h=H)
        return output

    def Normal_head(self, feature, SH):
        _, _, H, _ = feature.shape
        feature_flat = rearrange(feature.clone(), "b c h w -> b (h w) c")

        update = self.CA(feature_flat, SH)
        feature_flat = feature_flat + update
        feature_flat = feature_flat + self.MLP_normal(feature_flat.clone())
        output = self.proj_output_normal(feature_flat)
        output = rearrange(output, "b (h w) c -> b c h w", h=H)
        output = output / torch.norm(output, dim=1).unsqueeze(1)
        return output

    def forward(self, SH_query, features):
        SH_query = self.SH_patch_emb(SH_query).flatten(2)

        feature_0 = self.F_patch_emb0(features[0]).flatten(2) + self.PE
        SH_out1 = self.Cross_Attn(SH_query, feature_0)

        feature_1 = self.F_patch_emb1(features[1]).flatten(2) + self.PE
        F_out1 = self.Cross_Attn(feature_1, SH_out1)
        SH_out2 = self.Cross_Attn(SH_out1, F_out1)

        feature_2 = self.F_patch_emb2(features[2]).flatten(2) + self.PE
        F_out2 = self.Cross_Attn(feature_2, SH_out2)
        SH_out3 = self.Cross_Attn(SH_out2, F_out2)

        pixel_feature_depth = self.conv_1x1_depth(features[2])
        pixel_feature_normal = self.conv_1x1_normal(features[2])

        depth_index_map = self.cal_sim(SH_query, pixel_feature_depth)
        normal_index_map = self.cal_sim(SH_query, pixel_feature_normal)

        depth_SH_feature = torch.einsum("bnc, bnhw -> bchw", SH_query, depth_index_map)
        normal_SH_feature = torch.einsum("bnc, bnhw -> bchw", SH_query, normal_index_map)

        pixel_feature_depth *= depth_SH_feature
        pixel_feature_normal *= normal_SH_feature

        depth = self.Depth_head(pixel_feature_depth, SH_out3)
        normal = self.Normal_head(pixel_feature_normal, SH_out3)
        return depth, normal