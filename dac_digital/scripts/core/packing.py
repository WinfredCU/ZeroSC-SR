from __future__ import annotations
from typing import Dict, Any, Tuple
import math
import torch


def int_to_bits(x: torch.Tensor, num_bits: int, msb_first: bool = True) -> torch.Tensor:
    # x: [...], returns [..., num_bits]
    bits = [(x >> i) & 1 for i in range(num_bits)]
    if msb_first:
        bits = bits[::-1]
    out = torch.stack(bits, dim=-1).to(torch.uint8)
    return out


def bits_to_int(bits: torch.Tensor, msb_first: bool = True) -> torch.Tensor:
    # bits: [..., num_bits] in {0,1}
    num_bits = bits.size(-1)
    if msb_first:
        order = range(num_bits - 1, -1, -1)
    else:
        order = range(num_bits)
    x = torch.zeros(bits.shape[:-1], dtype=torch.long, device=bits.device)
    for i, b in enumerate(order):
        x = (x << 1) | (bits[..., b].long() & 1)
    return x


def pack_indices(
    codes: torch.Tensor,
    codebook_size: int,
    flatten_order: str = "C_then_T_time_fastest",
    bit_order: str = "msb_first",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    codes: [C, T] int
    Returns flat bitstream [B] in {0,1} and metadata for unpacking.
    Spec:
    - bits per index = ceil(log2(codebook_size))
    - MSB-first within each index when bit_order=="msb_first"
    - Flatten order: codebook-major then time, with time dimension fastest
    """
    assert codes.dim() == 2
    C, T = codes.shape
    num_bits = math.ceil(math.log2(codebook_size))
    bits = int_to_bits(codes, num_bits, msb_first=(bit_order == "msb_first"))  # [C, T, num_bits]
    # time fastest: [..., T, num_bits] → flatten T then num_bits within each index
    flat = bits.reshape(C * T * num_bits)
    meta = {
        "C": int(C),
        "T": int(T),
        "num_bits": int(num_bits),
        "codebook_size": int(codebook_size),
        "flatten_order": flatten_order,
        "bit_order": bit_order,
    }
    return flat.to(torch.uint8), meta


def unpack_indices(
    flat_bits: torch.Tensor,
    meta: Dict[str, Any],
) -> torch.Tensor:
    C = int(meta["C"])
    T = int(meta["T"])
    num_bits = int(meta["num_bits"])
    bit_order = meta.get("bit_order", "msb_first")
    bits = flat_bits.view(C, T, num_bits)
    codes = bits_to_int(bits, msb_first=(bit_order == "msb_first"))  # [C, T]
    return codes


def pad_to_multiple(bits: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, int]:
    mod = int(bits.numel() % multiple)
    if mod == 0:
        return bits, 0
    pad = multiple - mod
    padded = torch.cat([bits, torch.zeros(pad, dtype=bits.dtype, device=bits.device)], dim=0)
    return padded, pad
