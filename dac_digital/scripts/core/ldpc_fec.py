from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class LDPCSystem:
    H: np.ndarray
    G: np.ndarray
    k: int
    n: int
    max_iter: int
    matrix_id: str


def build_ldpc(k: int, n: int, dv: int, dc: int, max_iter: int, matrix_id: str, seed: int) -> LDPCSystem:
    import pyldpc
    H, G = pyldpc.make_ldpc(n, dv, dc, systematic=True, seed=seed)
    # Attempt to reduce to target (k, n) if generated dims mismatch
    # pyldpc returns H (m x n) and G (k x n)
    k_gen = G.shape[0]
    n_gen = G.shape[1]
    if k_gen != k or n_gen != n:
        # If different, we still proceed with generated (more stable) dimensions
        k, n = k_gen, n_gen
    return LDPCSystem(H=H, G=G, k=k, n=n, max_iter=max_iter, matrix_id=matrix_id)


def segment_bits(bits: np.ndarray, k: int) -> Tuple[np.ndarray, int]:
    total = bits.size
    num_blocks = int(np.ceil(total / k))
    pad_bits = num_blocks * k - total
    if pad_bits > 0:
        bits = np.concatenate([bits, np.zeros(pad_bits, dtype=np.uint8)])
    blocks = bits.reshape(num_blocks, k)
    return blocks, pad_bits


def encode_blocks(blocks: np.ndarray, ldpc: LDPCSystem) -> np.ndarray:
    import pyldpc
    coded = []
    for u in blocks:
        v = pyldpc.encode(ldpc.G, u)
        coded.append(v.astype(np.uint8))
    coded = np.stack(coded, axis=0)
    return coded


def decode_blocks(y_blocks: np.ndarray, snr_db: float, ldpc: LDPCSystem) -> np.ndarray:
    import pyldpc
    d_blocks = []
    for y in y_blocks:
        # y: real-valued observations (BPSK with noise)
        d = pyldpc.decode(ldpc.H, y, snr_db, ldpc.max_iter)
        u_hat = pyldpc.get_message(ldpc.G, d)
        d_blocks.append(u_hat.astype(np.uint8))
    return np.stack(d_blocks, axis=0)


def bits_to_bpsk(bits: np.ndarray) -> np.ndarray:
    return 1.0 - 2.0 * bits.astype(np.float64)


def bpsk_to_llr(y: np.ndarray, sigma2: float) -> np.ndarray:
    # LLR = 2*y/sigma2 for BPSK
    return 2.0 * y / float(sigma2)


def llr_to_bpsk_obs(llr: np.ndarray, sigma2: float) -> np.ndarray:
    # Invert LLR relation: y = 0.5 * LLR * sigma2
    return 0.5 * llr * float(sigma2)
