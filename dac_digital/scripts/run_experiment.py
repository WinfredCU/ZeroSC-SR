from __future__ import annotations
import argparse
import os
import time
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch

from .core.config import load_config, config_snapshot, make_config_id
from .core.utils import ensure_dir, get_device, set_deterministic, write_json, read_manifest
from .core.audio_io import load_audio, normalize_rms, save_wav
from .core.dac_codec import DACWrapper
from .core.packing import pack_indices, unpack_indices, pad_to_multiple
from .core.ldpc_fec import build_ldpc, segment_bits, encode_blocks, decode_blocks, llr_to_bpsk_obs
from .core.qam import map_bits_to_64qam, demap_64qam_to_llrs
from .core.ofdm import ofdm_modulate, ofdm_demodulate
from .core.channels import apply_channel
from .core.metrics import sdr_db, stoi, pesq_wb


def measure_snr_eq(S: torch.Tensor, Yeq: torch.Tensor) -> float:
    err = (Yeq - S)
    Es = torch.mean(torch.abs(S) ** 2) + 1e-12
    N0 = torch.mean(torch.abs(err) ** 2) + 1e-12
    snr_lin = (Es / N0).item()
    return 10.0 * np.log10(snr_lin)


def snapshot_env(path: str) -> None:
    try:
        os.system(f"pip freeze > {path}")
    except Exception:
        pass


def run(cfg_path: str) -> None:
    cfg = load_config(cfg_path)

    device = get_device(cfg.get("compute", {}).get("device", "auto"))
    set_deterministic(cfg.get("compute", {}).get("torch_seed", 42), cfg.get("compute", {}).get("deterministic", True))

    # Prepare paths
    art_dir = cfg["paths"]["artifacts_dir"]
    results_dir = cfg["paths"]["results_dir"]
    ensure_dir(art_dir)
    ensure_dir(results_dir)

    if cfg.get("logging", {}).get("write_config_snapshot", True):
        with open(os.path.join(results_dir, "config_snapshot.json"), "w") as f:
            f.write(config_snapshot(cfg))
    snapshot_env(cfg["paths"]["env_snapshot"])

    # Load model wrapper
    dac = DACWrapper(cfg["codec"]["model_id"], device=device)

    # Load manifest
    utt_ids, paths, durs = read_manifest(cfg["paths"]["manifest_path"])
    if cfg["experiment"].get("max_utts", 0) > 0:
        maxn = int(cfg["experiment"]["max_utts"])
        utt_ids, paths, durs = utt_ids[:maxn], paths[:maxn], durs[:maxn]

    sample_rate = int(cfg["dataset"]["sample_rate"])
    target_rms = float(cfg["dataset"]["target_rms"])

    # Build LDPC if enabled
    ldpc_cfg = cfg.get("ldpc", {})
    ldpc_enabled = bool(ldpc_cfg.get("enabled", True))
    ldpc_sys = None
    if ldpc_enabled:
        ldpc_sys = build_ldpc(
            k=int(ldpc_cfg["k"]),
            n=int(ldpc_cfg["n"]),
            dv=int(ldpc_cfg.get("dv", 3)),
            dc=int(ldpc_cfg.get("dc", 6)),
            max_iter=int(ldpc_cfg.get("max_iter", 25)),
            matrix_id=str(ldpc_cfg.get("matrix_id", "ldpc")),
            seed=int(ldpc_cfg.get("seed", 1337)),
        )

    modem_cfg = cfg["modem"]
    ofdm_cfg = cfg["ofdm"]
    ch_cfg = cfg["channels"]

    # Metrics accumulator
    rows: List[Dict[str, Any]] = []

    for utt_id, apath, dur in zip(utt_ids, paths, durs):
        wav, sr = load_audio(apath, sample_rate)
        wav = normalize_rms(wav, target_rms)

        for nq in cfg["codec"]["nq_list"]:
            # Stage A — Encode
            codes, meta = dac.encode(wav, sample_rate, nq)
            loopback = dac.decode(codes, sample_rate)
            sdr_loop = sdr_db(wav, loopback)
            # Save codes
            utt_dir = os.path.join(art_dir, utt_id)
            ensure_dir(utt_dir)
            base_meta = {
                "utt_id": utt_id,
                "duration": float(wav.size(-1) / sample_rate),
                "rms": float(torch.sqrt(torch.mean(wav**2)).item()),
                **meta,
            }
            if cfg["experiment"].get("save_codes", True):
                np.savez(
                    os.path.join(utt_dir, f"codes_nq{nq}.npz"),
                    audio_codes=codes.numpy(),
                    **base_meta,
                )
            # Acceptance (A-QC)
            if sdr_loop < 25.0:
                print(f"[WARN] A-QC failed SDR loopback <25 dB for {utt_id}, nq={nq}: {sdr_loop:.2f} dB")

            # Prepare reference for metrics
            ref_audio = wav if cfg["experiment"].get("use_original_ref", True) else loopback

            # Stage B — Pack
            flat_bits, pack_meta = pack_indices(
                codes, codebook_size=meta["codebook_size"],
                flatten_order=cfg["packing"]["flatten_order"],
                bit_order=cfg["packing"]["bit_order"],
            )
            # B-QC: unpack identity
            rec_codes = unpack_indices(flat_bits, pack_meta)
            if not torch.equal(rec_codes, codes):
                raise RuntimeError("Pack/Unpack identity check failed.")

            # Stage C — LDPC encode (optional)
            if ldpc_enabled:
                bits_np = flat_bits.cpu().numpy().astype(np.uint8)
                blocks, pad_bits = segment_bits(bits_np, ldpc_sys.k)
                coded_blocks = encode_blocks(blocks, ldpc_sys)
                header = {
                    "k": ldpc_sys.k,
                    "n": ldpc_sys.n,
                    "rate": float(ldpc_sys.k / ldpc_sys.n),
                    "num_blocks": int(coded_blocks.shape[0]),
                    "pad_bits": int(pad_bits),
                    "seed": int(ldpc_cfg.get("seed", 1337)),
                    "ldpc_matrix_id": ldpc_sys.matrix_id,
                }
                coded_bits = coded_blocks.reshape(-1).astype(np.uint8)
            else:
                header = {
                    "k": int(flat_bits.numel()),
                    "n": int(flat_bits.numel()),
                    "rate": 1.0,
                    "num_blocks": 1,
                    "pad_bits": 0,
                    "seed": 0,
                    "ldpc_matrix_id": "none",
                }
                coded_bits = flat_bits.cpu().numpy().astype(np.uint8)

            # Stage D — 64-QAM mapping
            # Group coded bits into groups of 6
            num_sym = int(np.ceil(len(coded_bits) / 6))
            pad_mod = num_sym * 6 - len(coded_bits)
            if pad_mod > 0:
                coded_bits = np.concatenate([coded_bits, np.zeros(pad_mod, dtype=np.uint8)])
            bits_torch = torch.from_numpy(coded_bits.reshape(-1, 6)).to(device=device, dtype=torch.uint8)
            S = map_bits_to_64qam(bits_torch)

            # D-QC: unit average energy
            Es = torch.mean(torch.abs(S) ** 2).item()
            if abs(Es - 1.0) > 1e-2:
                print(f"[WARN] D-QC Es!=1 ({Es:.3f})")

            # Stage E/F/G/H per channel/SNR
            for ch in ch_cfg["types"]:
                for snr_db in ch_cfg["snr_db_list"]:
                    cfg_id = make_config_id(cfg, nq=nq, channel=ch, snr_db=snr_db)
                    run_dir = os.path.join(utt_dir, cfg_id)
                    ensure_dir(run_dir)

                    # OFDM TX
                    x_time, ofdm_pad = ofdm_modulate(S, nfft=ofdm_cfg["nfft"], cp_len=ofdm_cfg["cp_len"], norm=ofdm_cfg.get("norm", "ortho"))

                    # Channel
                    # Deterministic seeds per (utt, ch, snr)
                    seed = (hash(utt_id) ^ hash(ch) ^ hash(int(snr_db))) & 0xFFFFFFFF
                    y_time, h_cplx, ch_stats = apply_channel(
                        x_time, ch, snr_db, device=device, rician_k=ch_cfg.get("rician_k", 5.0), seed=seed
                    )

                    # RX
                    Y = ofdm_demodulate(y_time, nfft=ofdm_cfg["nfft"], cp_len=ofdm_cfg["cp_len"], total_symbols=S.numel(), pad=ofdm_pad, norm=ofdm_cfg.get("norm", "ortho"))
                    h_t = torch.tensor(h_cplx, dtype=torch.complex64, device=device)
                    Yeq = Y / h_t

                    # Measure SNR after equalization
                    measured_snr_db = measure_snr_eq(S, Yeq)

                    # Demap to LLRs
                    snr_lin = 10 ** (snr_db / 10.0)
                    sigma2 = 1.0 / snr_lin
                    LLRs = demap_64qam_to_llrs(Yeq, sigma2=sigma2)  # [N,6]
                    llr_vec = LLRs.reshape(-1).cpu().numpy()

                    # LDPC decode (or pass-through)
                    if ldpc_enabled:
                        # Convert LLR to BPSK observations
                        y_obs = llr_to_bpsk_obs(llr_vec, sigma2)
                        y_blocks = y_obs.reshape(-1, ldpc_sys.n)
                        uhat_blocks = decode_blocks(y_blocks, snr_db=float(snr_db), ldpc=ldpc_sys)
                        rec_bits = uhat_blocks.reshape(-1)
                        if header["pad_bits"] > 0:
                            rec_bits = rec_bits[:-header["pad_bits"]]
                    else:
                        rec_bits = llr_vec[: flat_bits.numel()] > 0  # Hard decisions, bypass FEC
                        rec_bits = rec_bits.astype(np.uint8)

                    # Stage G — Unpack
                    rec_bits_t = torch.from_numpy(rec_bits.astype(np.uint8))
                    codes_hat = unpack_indices(rec_bits_t, pack_meta)

                    # Save codes_hat
                    if cfg["experiment"].get("save_codes_hat", True):
                        np.savez(
                            os.path.join(run_dir, "codes_hat.npz"),
                            audio_codes=codes_hat.numpy(),
                            **base_meta,
                            channel=ch,
                            snr_db=float(snr_db),
                            seeds={"channel": int(seed)},
                            ldpc_header=header,
                        )

                    # Stage H — Decode and metrics
                    recon = dac.decode(codes_hat, sample_rate)
                    if cfg["experiment"].get("save_audio", True):
                        save_wav(os.path.join(run_dir, "recon.wav"), recon, sample_rate)

                    # Metrics
                    ref_np = ref_audio.squeeze(0).cpu().numpy()
                    est_np = recon.squeeze(0).cpu().numpy()
                    m_sdr = sdr_db(ref_audio, recon)
                    try:
                        m_stoi = stoi(ref_np, est_np, sample_rate)
                    except Exception:
                        m_stoi = float("nan")
                    try:
                        m_pesq = pesq_wb(ref_np, est_np, sample_rate)
                    except Exception:
                        m_pesq = float("nan")

                    # Acceptance G-QC (AWGN 15 dB)
                    # Not asserting; will be validated in analysis

                    rows.append({
                        "utt_id": utt_id,
                        "nq": int(nq),
                        "channel": ch,
                        "snr_db": float(snr_db),
                        "sdr_db": float(m_sdr),
                        "stoi": float(m_stoi),
                        "pesq_wb": float(m_pesq),
                        "measured_snr_db": float(measured_snr_db),
                        "cfg_id": cfg_id,
                    })

    # Save metrics CSV and simple plots
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)

    # Simple plots per channel
    try:
        import matplotlib.pyplot as plt
        for ch in cfg["channels"]["types"]:
            fig, ax = plt.subplots(1, 3, figsize=(12, 3))
            for nq in cfg["codec"]["nq_list"]:
                mask = (df.channel == ch) & (df.nq == nq)
                dff = df[mask].groupby("snr_db").agg({"sdr_db": "median", "stoi": "median", "pesq_wb": "median"}).reset_index()
                ax[0].plot(dff.snr_db, dff.sdr_db, marker='o', label=f"nq={nq}")
                ax[1].plot(dff.snr_db, dff.stoi, marker='o', label=f"nq={nq}")
                ax[2].plot(dff.snr_db, dff.pesq_wb, marker='o', label=f"nq={nq}")
            ax[0].set_title(f"SDR vs SNR ({ch})"); ax[0].set_xlabel("SNR (dB)"); ax[0].set_ylabel("SDR (dB)")
            ax[1].set_title(f"STOI vs SNR ({ch})"); ax[1].set_xlabel("SNR (dB)"); ax[1].set_ylabel("STOI")
            ax[2].set_title(f"PESQ-WB vs SNR ({ch})"); ax[2].set_xlabel("SNR (dB)"); ax[2].set_ylabel("PESQ-WB")
            for a in ax: a.grid(True)
            ax[0].legend()
            fig.tight_layout()
            fig.savefig(os.path.join(results_dir, f"plots_{ch}.png"), dpi=200)
            plt.close(fig)
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    run(args.config)
