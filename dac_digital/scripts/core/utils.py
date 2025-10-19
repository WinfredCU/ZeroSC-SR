from __future__ import annotations
import os
import random
import json
from typing import Dict, Any, Tuple
import numpy as np
import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_device(pref: str = "auto") -> torch.device:
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_deterministic(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_manifest(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # returns arrays of utt_id (str), path (str), dur (float)
    utt_ids, paths, durs = [], [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if parts[0] == "utt_id":
                continue
            utt_ids.append(parts[0])
            paths.append(parts[1])
            durs.append(float(parts[2]))
    return np.array(utt_ids), np.array(paths), np.array(durs, dtype=float)


def complex_noise(shape, sigma2: float, device: torch.device) -> torch.Tensor:
    std = float(np.sqrt(sigma2 / 2.0))
    n = torch.randn(*shape, device=device) * std + 1j * (torch.randn(*shape, device=device) * std)
    return n
