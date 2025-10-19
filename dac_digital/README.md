# DAC-Digital Baseline (PyTorch)

This repository provides a fully digital semantic speech baseline that mirrors the Hybrid-DeepSCS setup:

DAC (16 kHz) → discrete indices → bit-packing → LDPC(1/2) → 64-QAM → OFDM → {AWGN, Rayleigh, Rician K=5} → demap/LDPC-decode → indices → DAC-decode → SDR/STOI/PESQ.

## Environment
- Python 3.10
- PyTorch + torchaudio (CPU OK; GPU optional)
- Transformers (for DAC encode/decode), Hugging Face Hub
- Metrics: pystoi, pesq
- LDPC: pyldpc (NumPy)

Recommended install (CPU):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Record exact versions for reproducibility (automatically saved to `artifacts/ENV.txt` by the runner):

```bash
pip freeze > dac_digital/artifacts/ENV.txt
```

## Repository Layout
```
dac_digital/
  cfg/
    experiment.yaml         # single source of truth for parameters
  data/
    manifest.csv            # utt_id, path, dur_s
  artifacts/
    <utt_id>/<config_id>/   # codes.npz, codes_hat.npz, recon.wav, metrics.json
  docs/
    methodology.md          # rationale & assumptions
    results/                # final tables, figures
  scripts/                  # runner scripts and core modules
  README.md                 # runbook + environment
  requirements.txt          # minimal pinned deps
```

## Data & Normalization
- Dataset: ≥50 utterances, ~10 s each (LibriSpeech/Libri-Light preferred), mono.
- Resample to 16 kHz; RMS-normalize to target RMS in config.
- `manifest.csv` columns: `utt_id,path,dur_s`.
- QC: duration in [9.5, 10.5] s; no clipping (|x| ≤ 1).

## Running the experiment
Edit the config in `dac_digital/cfg/experiment.yaml` as needed, then run:

```bash
python -m dac_digital.scripts.run_experiment --config dac_digital/cfg/experiment.yaml
```

Artifacts per (utt, config_id, channel, snr) will be saved under `dac_digital/artifacts/`.

## Acceptance checks
- Stage A (loopback): SDR > 25 dB.
- Stage B: pack→unpack identity.
- Stage C: deterministic LDPC output for fixed seeds/matrix.
- Stage D: 64-QAM Es normalization = 1.
- Stage E: OFDM identity channel, exact recovery.
- Stage F: Logged target SNR vs measured Es/N0; channel stats.
- Stage G: AWGN 15 dB: bit-perfect or BER < 1e-6.
- Stage H: Monotone metrics vs SNR; AWGN > Rician(K=5) > Rayleigh.

## Notes
- Perfect CSI equalization for flat channels.
- Gray mapping for 64-QAM; Es normalization = 1.
- Fixed frame duration and total TX energy across systems.
- Reproducibility: config snapshot + seeds logged per run.
