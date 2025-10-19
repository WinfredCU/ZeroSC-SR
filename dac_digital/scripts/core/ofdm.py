from __future__ import annotations
from typing import Tuple
import torch


def ofdm_modulate(symbols: torch.Tensor, nfft: int, cp_len: int, norm: str = "ortho") -> Tuple[torch.Tensor, int]:
    """
    symbols: [N] complex
    Returns time-domain signal [T] complex and number of pad symbols
    """
    N = symbols.numel()
    num_sym = int((N + nfft - 1) // nfft)
    pad = num_sym * nfft - N
    if pad > 0:
        symbols = torch.cat([symbols, torch.zeros(pad, dtype=symbols.dtype, device=symbols.device)], dim=0)
    X = symbols.view(num_sym, nfft)
    x_time = torch.fft.ifft(X, n=nfft, dim=1, norm=norm)  # [num_sym, nfft]
    cp = x_time[:, -cp_len:]
    x_cp = torch.cat([cp, x_time], dim=1)  # [num_sym, cp+nfft]
    y = x_cp.reshape(-1)
    return y, pad


def ofdm_demodulate(y: torch.Tensor, nfft: int, cp_len: int, total_symbols: int, pad: int, norm: str = "ortho") -> torch.Tensor:
    # y: [num_sym*(cp+nfft)] complex
    num_sym = int(len(y) // (nfft + cp_len))
    y = y.view(num_sym, nfft + cp_len)
    y_no_cp = y[:, cp_len:]
    Y = torch.fft.fft(y_no_cp, n=nfft, dim=1, norm=norm)
    S = Y.reshape(-1)
    if pad > 0:
        S = S[:-pad]
    if total_symbols is not None and S.numel() > total_symbols:
        S = S[:total_symbols]
    return S
