## Methodology

This baseline implements a fully digital semantic speech system using an off-the-shelf neural audio codec (DAC 16 kHz) to produce discrete indices, a documented bit packing spec, rate-1/2 LDPC forward error correction, 64-QAM with Gray mapping, OFDM framing, and flat channels (AWGN, Rayleigh, Rician K=5). The receiver demaps to LLRs, decodes LDPC, unpacks indices, and decodes with the same DAC. Metrics SDR, STOI, PESQ-WB are computed versus the original audio (or loopback) per configuration.

Assumptions:
- Perfect CSI equalization (flat fading).
- Fixed total TX energy per frame; 64-QAM normalized to Es=1.
- Reproducibility with fixed seeds per (utt, channel, SNR).
- Gray mapping for 64-QAM.

Risks & Mitigations:
- LDPC integration: use `pyldpc` NumPy implementation; deterministic matrices saved by seed and matrix_id.
- Packing mistakes: strict MSB-first, codebook-major flattening, unit tests pack↔unpack.
- Metric variance: ≥50 utterances and report medians/IQRs; RMS normalization.
