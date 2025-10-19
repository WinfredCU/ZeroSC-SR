from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
import torch
from .utils import complex_noise


def apply_channel(
    x: torch.Tensor,
    channel: str,
    snr_db: float,
    device: torch.device,
    rician_k: float = 5.0,
    seed: int = 0,
) -> Tuple[torch.Tensor, complex, Dict[str, Any]]:
    """
    x: time-domain complex signal [T]
    Returns y, h (complex scalar), and stats
    SNR is interpreted as Es/N0 with Es=1 at the QAM symbol level and preserved through unitary OFDM.
    """
    rng = np.random.default_rng(seed)
    snr_lin = float(10 ** (snr_db / 10.0))
    sigma2 = 1.0 / snr_lin

    if channel.lower() == "awgn":
        h = 1.0 + 0.0j
    elif channel.lower() == "rayleigh":
        hr = rng.normal(0, np.sqrt(0.5))
        hi = rng.normal(0, np.sqrt(0.5))
        h = hr + 1j * hi
    elif channel.lower() == "rician":
        K = float(rician_k)
        mu = np.sqrt(K / (K + 1.0))
        sigma = np.sqrt(1.0 / (2.0 * (K + 1.0)))
        hr = rng.normal(mu, sigma)
        hi = rng.normal(0.0, sigma)
        h = hr + 1j * hi
    else:
        raise ValueError(f"Unknown channel: {channel}")

    h_torch = torch.tensor(h, dtype=torch.complex64, device=device)
    y = h_torch * x + complex_noise(x.shape, sigma2, device)

    stats = {
        "snr_db_target": snr_db,
        "snr_lin": snr_lin,
        "sigma2": sigma2,
        "E_h2": float((hr if 'hr' in locals() else 1.0) ** 2 + (hi if 'hi' in locals() else 0.0) ** 2)
        if channel.lower() != "awgn"
        else 1.0,
        "h_real": float(np.real(h)),
        "h_imag": float(np.imag(h)),
    }
    return y, h, stats
