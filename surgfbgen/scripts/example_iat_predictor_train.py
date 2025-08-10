#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a SIMULTANEOUS IAT predictor using PeskaVLP with VIDEO-ONLY inputs
(aux_text_names = []).
- Uniformly samples frames per clip and resizes to a fixed size for batching.
- Patches the model's forward() to skip text when aux_text_names is empty.
- Expects a CSV listing clips and label indices, and a JSON listing class names.

CSV format (example):
clip,instrument,action,tissue
case01_clip001.avi,3,5,2
...

JSON (class map) format:
{
  "instrument": ["...", "...", ...],
  "action":     ["...", "...", ...],
  "tissue":     ["...", "...", ...]
}

Usage:
python example_iat_predictor_train.py \
  --clips_dir /path/to/clips \
  --train_csv /path/to/train.csv \
  --val_csv /path/to/val.csv \
  --output_dir /path/to/output_dir \
  --add_repo_dir_prefix \

Example usage:
python example_iat_predictor_train.py \
  --clips_dir /home/firdavs/surgery/clips_with_wiggle/fb_clips_wiggle \
  --train_csv data/iat_predictor_splits/train1.csv \
  --val_csv data/iat_predictor_splits/val1.csv \
  --output_dir outputs/example_iat_predictor_train \
  --add_repo_dir_prefix
"""

import os
import json
import argparse
import random
import types
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# --- Your project imports (adjust paths if your module layout differs) ---
from surgfbgen.logger.logger import get_logger
from surgfbgen.models.utils import load_surgical_VL_model
from surgfbgen.prompts.base import prompt_library  # not used here but keeps env consistent

# These classes/enums are assumed to be available from your IAT predictor module
from surgfbgen.models.iat_predictor import (
    IATPredictor,
    IATPredictorType,
    IATPredictorSurgicalVLConfig,
    IATPredictorSurgicalVLTrainConfig,
    PredictionType,
    ClassificationHeadConfig,
    FFNConfig,
)

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def uniform_indices(n_total: int, n_sample: int) -> List[int]:
    if n_total <= 0:
        return []
    if n_total <= n_sample:
        # repeat last frame to reach n_sample
        idxs = list(range(n_total)) + [n_total - 1] * (n_sample - n_total)
        return idxs
    step = n_total / float(n_sample)
    return [min(int(i * step), n_total - 1) for i in range(n_sample)]

def load_all_frames(path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for _ in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

# ----------------------------
# Dataset
# ----------------------------
class ClipsIATDataset(Dataset):
    """
    Returns:
      {
        'video_frames': torch.uint8 tensor [T, H, W, C],
        'text_inputs' : {
            'dialogue': str,
            'procedure': str,
            'task': str,
        },
        'labels'      : {'instrument': int, 'action': int, 'tissue': int}
      }
    """
    def __init__(
        self,
        csv_path: str,
        clips_dir: str,
        num_frames: int = 50,
        resize_hw: Tuple[int, int] = (224, 224),
        clip_ext: str = "avi",
    ):
        import pandas as pd  # local import to avoid global dependency
        self.df = pd.read_csv(csv_path)
        self.clips_dir = clips_dir
        self.num_frames = num_frames
        self.resize_hw = resize_hw
        self.clip_ext = clip_ext
        
        # required columns
        for col in ["cvid", "dialogue", "instrument", "action", "tissue", "procedure_defn", "task_defn"]:
            if col not in self.df.columns:
                raise ValueError(f"Missing column '{col}' in {csv_path}")

    def get_classes(self) -> Dict[str, List[str]]:
        """Get each unique class for I/A/T."""
        classes = {
            "instrument": self.df["instrument"].unique().tolist(),
            "action": self.df["action"].unique().tolist(),
            "tissue": self.df["tissue"].unique().tolist(),
        }
        return classes
    
    def get_class_counts(self) -> Dict[str, Dict[str, int]]:
        """Get counts of each class for I/A/T."""
        counts = {
            "instrument": self.df["instrument"].value_counts().to_dict(),
            "action": self.df["action"].value_counts().to_dict(),
            "tissue": self.df["tissue"].value_counts().to_dict(),
        }
        return counts
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        clip_name = row["cvid"]
        # Permit clip names with or without extension
        if not clip_name.lower().endswith(f".{self.clip_ext}"):
            clip_name = f"{clip_name}.{self.clip_ext}"
        clip_path = os.path.join(self.clips_dir, clip_name)

        # try to read frames; if it fails, return a sentinel sample
        try:
            frames = load_all_frames(clip_path)
        except Exception as e:
            return {
                "_invalid": True,
                "cvid": str(row["cvid"]),
                "error": f"{type(e).__name__}: {e}",
            }

        if len(frames) == 0:
            return {
                "_invalid": True,
                "cvid": str(row["cvid"]),
                "error": "empty_video",
            }


        idxs = uniform_indices(len(frames), self.num_frames)
        sampled = [frames[i] for i in idxs]

        # Resize to a fixed size so we can batch raw frames
        H, W = self.resize_hw
        sampled = [cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in sampled]

        # HWC uint8 -> torch
        vid = torch.from_numpy(np.stack(sampled, axis=0)).to(torch.uint8)  # [T,H,W,C], uint8

        labels = {
            "instrument": str(row["instrument"]),
            "action": str(row["action"]),
            "tissue": str(row["tissue"]),
        }
        text_inputs = {
            "procedure": str(row["procedure_defn"]),
            "task": str(row["task_defn"]),
        }
        return {
            "original_dialogue": str(row["dialogue"]),
            "cvid": str(row["cvid"]),
            "video_frames": vid,  # [T,H,W,C], uint8
            "text_inputs": text_inputs,
            "labels": labels,
            "_invalid": False,
        }

def make_one_hot_collate(all_classes: Dict[str, List[str]]):
    idx_maps = {k: {cls: i for i, cls in enumerate(v)} for k, v in all_classes.items()}
    sizes = {k: len(v) for k, v in all_classes.items()}

    def _collate(samples):
        # filter out invalid samples from dataset
        valid = [s for s in samples if isinstance(s, dict) and not s.get("_invalid", False)]
        dropped = [s.get("cvid", None) for s in samples if isinstance(s, dict) and s.get("_invalid", False)]

        if len(valid) == 0:
            # return a small sentinel batch the train/val loops can detect
            return {"_invalid_batch": True, "num_dropped": len(dropped), "dropped_cvids": dropped}

        vids = torch.stack([s["video_frames"] for s in valid], dim=0)  # [Bv,T,H,W,C], uint8
        texts = [s["text_inputs"] for s in valid]

        labels = {}
        for key in ["instrument", "action", "tissue"]:
            oh = torch.zeros(len(valid), sizes[key], dtype=torch.float32)
            for i, s in enumerate(valid):
                cls_name = s["labels"][key]
                oh[i, idx_maps[key][cls_name]] = 1.0
            labels[key] = oh

        return {
            "video_frames": vids,
            "text_inputs": texts,
            "labels": labels,
            "_invalid_batch": False,
            "num_dropped": len(dropped),
            "dropped_cvids": dropped[:8],  # cap the list so logs don't explode
        }

    return _collate

# ----------------------------
# Build configs
# ----------------------------
def make_configs(
    class_map: Dict[str, List[str]],
    output_dir: str,
    hidden_sizes: List[int] = (128, 64),
    dropout: float = 0.1,
):
    # validate and build head configs
    iat_clf_head_cfgs = {
        k: ClassificationHeadConfig(num_classes=len(v), dropout=dropout)
        for k, v in class_map.items()
    }

    ffn_cfg = FFNConfig(hidden_sizes=list(hidden_sizes), activation="ReLU", dropout=dropout)

    model_cfg = IATPredictorSurgicalVLConfig(
        type=IATPredictorType.SURGICAL_VL,
        prediction_type=PredictionType.SIMULTANEOUS,
        surgical_VL_model_name="PeskaVLP",
        aux_text_names=[],  # << video-only
        ffn_config=ffn_cfg,
        IAT_class_names=class_map,
        IAT_clf_head_configs=iat_clf_head_cfgs,
    )

    train_cfg = IATPredictorSurgicalVLTrainConfig(
        optimizer_class="AdamW",
        initial_learning_rate=1e-4,
        learning_rate_type="cosine",
        warmup_ratio=0.1,
        max_iters=5000,           # adjust as needed
        n_iter_no_change=500,    # early stop patience (in val checks)
        loss_fn="CrossEntropyLoss",
        output_dir=output_dir,
        freeze_surgical_vl=True,  # fine-tune heads/ffn only
        val_interval=100
    )
    return model_cfg, train_cfg

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_dir", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--add_repo_dir_prefix", action="store_true")

    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--ext", type=str, default="avi")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    if args.add_repo_dir_prefix:
        args.clips_dir = os.path.join(os.environ['REPO_DIRECTORY'], args.clips_dir)
        args.train_csv = os.path.join(os.environ['REPO_DIRECTORY'], args.train_csv)
        args.val_csv = os.path.join(os.environ['REPO_DIRECTORY'], args.val_csv)
        args.output_dir = os.path.join(os.environ['REPO_DIRECTORY'], args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "train.log")
    logger = get_logger("example_iat_predictor_train", log_file=log_file, console_output=True)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Datasets / loaders
    train_ds = ClipsIATDataset(
        csv_path=args.train_csv,
        clips_dir=args.clips_dir,
        num_frames=args.num_frames,
        resize_hw=(args.height, args.width),
        clip_ext=args.ext,
    )
    val_ds = ClipsIATDataset(
        csv_path=args.val_csv,
        clips_dir=args.clips_dir,
        num_frames=args.num_frames,
        resize_hw=(args.height, args.width),
        clip_ext=args.ext,
    )
    
    # Load class map
    train_classes = train_ds.get_classes()
    val_classse = val_ds.get_classes()
    # {instrument, action, tissue}: set of all classes
    all_classes = {k: list(set(train_classes.get(k, []) + val_classse.get(k, []))) for k in set(train_classes) | set(val_classse)}
    logger.info(f"IAT classes:")
    logger.info(json.dumps(all_classes, indent=4))

    collate_fn = make_one_hot_collate(all_classes)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        collate_fn=collate_fn,
        drop_last=False,
    )
    logger.info(f"Train iters/epoch (approx): {len(train_loader)}; Val batches: {len(val_loader)}")

    # Configs
    model_cfg, train_cfg = make_configs(all_classes, args.output_dir)

    # Build predictor (this will load PeskaVLP via your utils)
    predictor = IATPredictor(iat_predictor_config=model_cfg, device=device, verbose=True, logger=logger)

    # Train
    predictor.train(train_cfg, train_loader, val_loader)
    logger.info("Training finished.")

if __name__ == "__main__":
    main()
