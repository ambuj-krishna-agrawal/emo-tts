#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# evaluate_emotion.py
# -----------------------------------------------------------------------------
# Compute emotion‑aware similarity between reference and synthesised audio:
#   1) Emo2Vec cosine distance (continuous)
#   2) SpeechBrain wav2vec2‑IEMOCAP classifier accuracy (discrete)
# -----------------------------------------------------------------------------
from __future__ import annotations
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm

from utils.metrics import load_emo2vec

# try SpeechBrain import gracefully
try:
    from speechbrain.pretrained import EncoderClassifier
    SB_AVAILABLE = True
    SB_MODEL_TAG = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
except Exception as e:
    SB_AVAILABLE = False
    logging.warning("SpeechBrain not available: %s", e)

# ───────────────────────────── Config dataclass ──────────────────────────────
@dataclass
class EvalConfig:
    ref_dir: Path
    syn_dir: Path
    sample_rate: int = 16_000
    metrics: List[str] = None  # ["emo2vec", "speechbrain"]

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["emo2vec", "speechbrain"]

# ────────────────────────────────── Helpers ──────────────────────────────────
def _load_wav(path: Path, sr: int) -> torch.Tensor:
    wav, file_sr = sf.read(path)
    if file_sr != sr:
        raise ValueError(f"SR mismatch: {file_sr} != {sr} ({path})")
    return torch.tensor(wav, dtype=torch.float32)

def evaluate(cfg: EvalConfig):
    logging.info("Eval config:\n%s", cfg)

    if "emo2vec" in cfg.metrics:
        emo2vec = load_emo2vec()
        logging.info("Loaded Emo2Vec.")

    if "speechbrain" in cfg.metrics and SB_AVAILABLE:
        sb_clf = EncoderClassifier.from_hparams(source=SB_MODEL_TAG)
        logging.info("Loaded SpeechBrain classifier.")
    elif "speechbrain" in cfg.metrics:
        logging.warning("SpeechBrain metric requested but not available.")
        cfg.metrics.remove("speechbrain")

    cos_dists = []
    sb_correct = 0
    sb_total   = 0

    ref_files = sorted(cfg.ref_dir.glob("*.wav"))
    for ref_path in tqdm(ref_files, desc="Scoring"):
        utt_id   = ref_path.stem
        syn_path = cfg.syn_dir / f"{utt_id}.wav"
        if not syn_path.exists():
            logging.debug("Skip (no synth): %s", utt_id)
            continue

        ref_wav = _load_wav(ref_path, cfg.sample_rate)
        syn_wav = _load_wav(syn_path, cfg.sample_rate)

        # ── Emo2Vec cosine distance ─────────────────────────────────────────
        if "emo2vec" in cfg.metrics:
            ref_emb = emo2vec(ref_wav, cfg.sample_rate)
            syn_emb = emo2vec(syn_wav, cfg.sample_rate)
            cos = torch.nn.functional.cosine_similarity(
                ref_emb, syn_emb, dim=0, eps=1e-8
            ).item()
            cos_dists.append(1 - cos)   # distance (lower is better)

        # ── SpeechBrain discrete accuracy ───────────────────────────────────
        if "speechbrain" in cfg.metrics:
            with torch.no_grad():
                ref_pred = sb_clf.classify_batch(ref_wav.unsqueeze(0))[3][0]
                syn_pred = sb_clf.classify_batch(syn_wav.unsqueeze(0))[3][0]
            sb_correct += int(ref_pred == syn_pred)
            sb_total   += 1

    # ───────────────────────────── Results ──────────────────────────────────
    if "emo2vec" in cfg.metrics and cos_dists:
        logging.info("Emo2Vec mean cosine distance: %.4f (↓)", np.mean(cos_dists))
    if "speechbrain" in cfg.metrics and sb_total:
        logging.info("SpeechBrain top‑1 accuracy : %.2f%% (↑)",
                     sb_correct / sb_total * 100)

# ─────────────────────────────────── Main ────────────────────────────────────
def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description="Emotion‑aware evaluation")
    p.add_argument("--ref_dir", type=Path, required=True,
                   help="Directory with reference WAVs")
    p.add_argument("--syn_dir", type=Path, required=True,
                   help="Directory with synthesised WAVs")
    p.add_argument("--sample_rate", type=int, default=16_000)
    p.add_argument("--metrics", nargs="+",
                   choices=["emo2vec", "speechbrain"], default=None)
    args = p.parse_args()
    return EvalConfig(**vars(args))

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = parse_args()
    evaluate(cfg)
