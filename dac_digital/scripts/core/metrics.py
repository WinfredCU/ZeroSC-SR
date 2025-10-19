from __future__ import annotations
from typing import Dict
import numpy as np
import torch
from pystoi import stoi as stoi_metric
from pesq import pesq as pesq_metric


def sdr_db(ref: torch.Tensor, est: torch.Tensor) -> float:
    ref = ref.view(-1).double()
    est = est.view(-1).double()
    num = torch.sum(ref ** 2).item() + 1e-12
    den = torch.sum((ref - est) ** 2).item() + 1e-12
    return 10.0 * np.log10(num / den)


def stoi(ref: np.ndarray, est: np.ndarray, sr: int) -> float:
    return float(stoi_metric(ref, est, sr, extended=False))


def pesq_wb(ref: np.ndarray, est: np.ndarray, sr: int) -> float:
    return float(pesq_metric(sr, ref, est, "wb"))
