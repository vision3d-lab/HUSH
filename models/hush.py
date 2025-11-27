import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Any, Dict, Optional, Tuple
from .layers import swin_large_22k, MSDA_PixelDecoder, SH_Coeff_Extractor, SH_Attention_Module
from .utils import build_basis


class HUSH(nn.Module):
    def __init__(self, min_val=0.1, max_val=10, degree=10):
        super(HUSH, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.degree = degree

        self.encoder = swin_large_22k(pretrained=True)
        self.decoder = MSDA_PixelDecoder()

        self.SH_basis = build_basis(degree, 1, [256, 512])
        self.SHCE = SH_Coeff_Extractor()
        self.SHAM = SH_Attention_Module()


    def invert_encoder_output_order(self, xs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        return tuple(xs[::-1])

    def filter_decoder_relevant_resolutions(self, decoder_outputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        return tuple(decoder_outputs[1:])

    def forward(self, x):
        encoder_output = self.encoder(x)
        encoder_output = self.invert_encoder_output_order(encoder_output)

        fpn_output, decoder_output = self.decoder(encoder_output)
        decoder_output = self.filter_decoder_relevant_resolutions(decoder_output)
        fpn_output = self.filter_decoder_relevant_resolutions(fpn_output)

        SH_coeffs = self.SHCE(decoder_output[-1])
        SH_query = torch.einsum("bn, nhw -> bnhw", SH_coeffs, (self.SH_basis).squeeze(0))
        
        depth, normal = self.SHAM(SH_query, fpn_output)
        return depth, normal
