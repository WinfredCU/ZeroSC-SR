from __future__ import annotations
from typing import Dict, Any, Tuple
import torch


class DACWrapper:
    def __init__(self, model_id: str, device: torch.device):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.codebook_size = 1024  # typical for DAC RVQ stages
        self._load_model()

    def _load_model(self) -> None:
        # Try different known APIs; raise clear error if unavailable
        try:
            # Descript DAC repository package
            from dac.model import DAC
            # Attempt common patterns for loading
            try:
                self.model = DAC.from_pretrained(self.model_id)
            except Exception:
                try:
                    self.model = DAC.load("16khz")
                except Exception as e2:
                    raise ImportError(
                        f"Failed to load DAC model '{self.model_id}'. Install 'dac' package and ensure compatibility. Original: {e2}"
                    )
            self.model.to(self.device)
            self.model.eval()
            # Try to read codebook size if exposed
            if hasattr(self.model, "quantizer") and hasattr(self.model.quantizer, "codebook_size"):
                self.codebook_size = int(self.model.quantizer.codebook_size)
        except ImportError as e:
            raise ImportError(
                "Descript DAC is required. Install with: pip install git+https://github.com/descriptinc/descript-audio-codec.git#egg=dac"
            ) from e

    @torch.no_grad()
    def encode(self, wav: torch.Tensor, sr: int, nq: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Returns:
            codes: LongTensor [nq, T_idx]
            meta: dict with fields: codebook_size, nq, sr, num_frames
        """
        wav = wav.to(self.device)
        # Attempt different method names; adjust to model API
        codes = None
        try:
            # Hypothetical API: returns list/stack per RVQ stage
            out = self.model.encode(wav, sample_rate=sr, num_codebooks=nq)
            # Normalize to tensor [nq, T]
            if isinstance(out, (list, tuple)):
                codes = torch.stack([o.squeeze(0).long() for o in out], dim=0)
            elif isinstance(out, torch.Tensor):
                codes = out.long()
        except Exception:
            try:
                out = self.model.encode(wav)
                if isinstance(out, (list, tuple)):
                    out = out[:nq]
                    codes = torch.stack([o.squeeze(0).long() for o in out], dim=0)
                else:
                    codes = out.long()
            except Exception as e:
                raise RuntimeError(
                    f"DAC encode failed; please verify API compatibility. Error: {e}"
                )
        if codes is None:
            raise RuntimeError("DAC encode did not return codes.")
        meta = {
            "codebook_size": int(self.codebook_size),
            "nq": int(nq),
            "sr": int(sr),
            "num_frames": int(codes.size(-1)),
        }
        return codes.cpu(), meta

    @torch.no_grad()
    def decode(self, codes: torch.Tensor, sr: int) -> torch.Tensor:
        # Accept [nq, T] codes
        codes = codes.to(self.device)
        wav = None
        try:
            wav = self.model.decode(codes)
        except Exception:
            try:
                # Some APIs expect list per stage
                stages = [codes[i].unsqueeze(0) for i in range(codes.size(0))]
                wav = self.model.decode(stages)
            except Exception as e:
                raise RuntimeError(
                    f"DAC decode failed; please verify API compatibility. Error: {e}"
                )
        # Expect shape [B, C, T] or [C, T]
        if wav.dim() == 3:
            wav = wav[0]
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav[:1]
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        return wav.cpu()
