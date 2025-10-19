from __future__ import annotations
from typing import Tuple
import torch
import numpy as np


def gray_code(n: int) -> torch.Tensor:
    x = torch.arange(2**n, dtype=torch.int64)
    return x ^ (x >> 1)


def pam8_levels() -> torch.Tensor:
    # Levels for 8-PAM: [-7, -5, -3, -1, 1, 3, 5, 7]
    return torch.tensor([-7, -5, -3, -1, 1, 3, 5, 7], dtype=torch.float32)


def normalization_factor_64qam() -> float:
    # Average symbol energy before normalization for 64-QAM
    # 8-PAM per axis avg E = 21, total Es = 42
    return float(np.sqrt(42.0))


def map_bits_to_64qam(bits: torch.Tensor) -> torch.Tensor:
    """
    bits: [N, 6] in {0,1}
    Returns complex symbols [N] normalized to Es=1.
    Gray mapping: first 3 bits → I, last 3 bits → Q
    """
    assert bits.dim() == 2 and bits.size(1) == 6
    device = bits.device
    gray = gray_code(3).to(device)
    # Map Gray code to index 0..7: Build inverse map from code->index
    inv_map = torch.empty(8, dtype=torch.long, device=device)
    inv_map[gray] = torch.arange(8, device=device)

    bI = bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]
    bQ = bits[:, 3] * 4 + bits[:, 4] * 2 + bits[:, 5]

    idxI = inv_map[bI]
    idxQ = inv_map[bQ]

    levels = pam8_levels().to(device)
    I = levels[idxI]
    Q = levels[idxQ]

    s = (I + 1j * Q) / normalization_factor_64qam()
    return s


def demap_64qam_to_llrs(y: torch.Tensor, sigma2: float) -> torch.Tensor:
    """
    y: [N] complex received after equalization
    Returns LLRs [N, 6] using exact LLR via log-sum-exp over 1D PAM for I and Q
    """
    device = y.device
    I = y.real
    Q = y.imag
    levels = pam8_levels().to(device)
    norm = normalization_factor_64qam()
    levels_n = levels / norm

    # Precompute Gray labels per level index
    gray = gray_code(3).to(device)
    labels = torch.stack([(gray >> 2) & 1, (gray >> 1) & 1, gray & 1], dim=1).to(torch.float32)

    def llr_axis(x: torch.Tensor) -> torch.Tensor:
        # x: [N]
        # For each bit position, compute LLR by log-sum-exp over levels with bit=0 vs bit=1
        N = x.numel()
        x_exp = x.view(N, 1)
        dist2 = (x_exp - levels_n.view(1, 8)) ** 2  # [N, 8]
        metric = -dist2 / float(sigma2)  # ignoring constant 1 factor
        # log-sum-exp per bit
        max_m = metric.max(dim=1, keepdim=True).values
        exp_m = torch.exp(metric - max_m)
        llrs = []
        for b in range(3):
            mask0 = (labels[:, b] == 0).view(1, 8)
            mask1 = ~mask0
            num0 = torch.log(torch.sum(exp_m * mask0, dim=1) + 1e-12)
            num1 = torch.log(torch.sum(exp_m * mask1, dim=1) + 1e-12)
            llr_b = (num0 - num1)
            llrs.append(llr_b)
        return torch.stack(llrs, dim=1)  # [N, 3]

    llrI = llr_axis(I)
    llrQ = llr_axis(Q)
    return torch.cat([llrI, llrQ], dim=1)
