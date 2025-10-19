from __future__ import annotations
import hashlib
import json
from typing import Any, Dict
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def config_snapshot(cfg: Dict[str, Any]) -> str:
    return json.dumps(cfg, sort_keys=True, indent=2)


def make_config_id(cfg: Dict[str, Any], nq: int, channel: str, snr_db: int) -> str:
    ldpc = cfg.get("ldpc", {})
    ofdm = cfg.get("ofdm", {})
    fec_tag = "ldpcK{}N{}".format(ldpc.get("k", 0), ldpc.get("n", 0)) if ldpc.get("enabled", True) else "nofec"
    base = f"dac16-nq{nq}-{fec_tag}-ofdm{ofdm.get('nfft',0)}cp{ofdm.get('cp_len',0)}-{channel}-{snr_db}dB"
    return base


def hash_small(d: Dict[str, Any]) -> str:
    j = json.dumps(d, sort_keys=True).encode()
    return hashlib.sha1(j).hexdigest()[:10]
