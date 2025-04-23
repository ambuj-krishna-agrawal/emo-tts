#!/usr/bin/env python3
# evaluate_metrics.py

import os
import re
from funasr import AutoModel
import torch
import torch.nn.functional as F

# ─── CONFIGURE PATHS HERE ──────────────────────────────────────────────────────
ground_truth_dir = '/data/user_data/ambuja/multidialog_sds_pairs/audio'
mistral_dir       = '/data/user_data/ambuja/multidialogue_emotivoice_out/mistral_7b_q4'
llama_dir         = '/data/user_data/ambuja/multidialogue_emotivoice_out/llama_3_3b_q4'
# ───────────────────────────────────────────────────────────────────────────────

# load the emotion2vec model once
model_id = "iic/emotion2vec_plus_large"
model = AutoModel(model=model_id, hub="hf", disable_update=True)

# regex helpers
def extract_gt_id(filename):
    """From 'pair_00142_B.wav' → '00142'"""
    m = re.search(r'pair_(\d+)_B\.wav', filename)
    return m.group(1) if m else None

def extract_model_id(filename):
    """From '00137_steered.wav' or '00137_neutral.wav' → '00137'"""
    m = re.search(r'(\d+)_(steered|neutral)\.wav', filename)
    return m.group(1) if m else None

def evaluate(model_dir, model_name):
    # collect file lists
    steered_files = [os.path.join(model_dir, f)
                     for f in os.listdir(model_dir)
                     if f.endswith('.wav') and 'steered' in f]
    neutral_files = [os.path.join(model_dir, f)
                     for f in os.listdir(model_dir)
                     if f.endswith('.wav') and 'neutral' in f]

    # map ground-truth IDs to their WAV paths
    gt_map = {}
    for f in os.listdir(ground_truth_dir):
        if f.endswith('_B.wav'):
            id_ = extract_gt_id(f)
            if id_:
                gt_map[id_] = os.path.join(ground_truth_dir, f)

    # map IDs → model WAVs
    steered_map = {extract_model_id(os.path.basename(p)): p for p in steered_files}
    neutral_map = {extract_model_id(os.path.basename(p)): p for p in neutral_files}

    steered_scores = []
    neutral_scores = []

    for utt_id, gt_path in gt_map.items():
        if utt_id not in steered_map or utt_id not in neutral_map:
            print(f"[{model_name}] Missing files for ID {utt_id}")
            continue

        try:
            # generate embeddings
            gt_out      = model.generate(gt_path,
                                        output_dir="./emo2vec_outputs",
                                        granularity="utterance",
                                        extract_embedding=True)
            steered_out = model.generate(steered_map[utt_id],
                                         output_dir="./emo2vec_outputs",
                                         granularity="utterance",
                                         extract_embedding=True)
            neutral_out = model.generate(neutral_map[utt_id],
                                         output_dir="./emo2vec_outputs",
                                         granularity="utterance",
                                         extract_embedding=True)

            gt_emb      = torch.from_numpy(gt_out[0]['feats'])
            steered_emb = torch.from_numpy(steered_out[0]['feats'])
            neutral_emb = torch.from_numpy(neutral_out[0]['feats'])

            # cosine similarities
            s_sim = F.cosine_similarity(gt_emb, steered_emb, dim=0).item()
            n_sim = F.cosine_similarity(gt_emb, neutral_emb, dim=0).item()

            steered_scores.append(s_sim)
            neutral_scores.append(n_sim)

            print(f"[{model_name}] ID {utt_id}: "
                  f"steered={s_sim:.4f}, neutral={n_sim:.4f}")

        except Exception as e:
            print(f"[{model_name}] Error ID {utt_id}: {e}")

    # summary stats
    if steered_scores:
        print(f"\n{model_name} Steered → "
              f"min={min(steered_scores):.4f}, "
              f"max={max(steered_scores):.4f}, "
              f"avg={sum(steered_scores)/len(steered_scores):.4f}")
    if neutral_scores:
        print(f"{model_name} Neutral → "
              f"min={min(neutral_scores):.4f}, "
              f"max={max(neutral_scores):.4f}, "
              f"avg={sum(neutral_scores)/len(neutral_scores):.4f}")
    print("="*60, "\n")


if __name__ == "__main__":
    evaluate(mistral_dir, "Mistral")
    evaluate(llama_dir,   "Llama")
