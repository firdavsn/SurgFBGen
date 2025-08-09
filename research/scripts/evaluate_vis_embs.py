import pandas as pd
import json
import numpy as np
import os
import torch
from PIL import Image
from mmengine.config import Config
from transformers import set_seed
import random
import cv2
import h5py
import logging
from logging import Logger
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import precision_recall_curve, roc_auc_score, accuracy_score

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOGGING_PATH = f'/home/firdavs/surgery/surgical_fb_generation/SurgicalFeedbackGeneration/logging/evaluate_vis_embs-dt=({datetime.now().strftime("%Y-%m-%d-%H-%M")}).log'
OUTPUT_DIR = '/home/firdavs/surgery/surgical_fb_generation/SurgicalFeedbackGeneration/outputs/evals/vision'
ALL_ANNOTATIONS_PATH = '/home/firdavs/surgery/surgical_fb_generation/SurgicalFeedbackGeneration/data/urology-related/annotations/cmb_all_mapped.csv'

def set_seed_all(seed, logger: Logger = None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)
    print(f"Set seed: {seed}")
    if logger:
        logger.info(f"Set seed: {seed}")

def setup_logger(log_file):
    """Configure and return a logger that writes to a file."""
    logger = logging.getLogger('VideoMAE_Logger')
    logger.setLevel(logging.INFO)  # Capture INFO level and above
    # Create file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    # Define log format: timestamp - level - message
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # Add handler to logger
    logger.addHandler(fh)
    return logger

def load_annotations(annotations_path):
    df = pd.read_csv(annotations_path)
    df = df['']

def main():
    logger = setup_logger(LOGGING_PATH)
    set_seed_all(SEED, logger)
    
    surgvlp_h5_path = os.path.join(OUTPUT_DIR, "surgvlp_fbk_vis_embs.h5")
    logger.info(f"surgvlp_h5_path: {surgvlp_h5_path}")
    print(f"surgvlp_h5_path: {surgvlp_h5_path}")
        
    hecvl_h5_path = os.path.join(OUTPUT_DIR, "hecvl_fbk_vis_embs.h5")
    logger.info(f"hecvl_h5_path: {hecvl_h5_path}")
    print(f"hecvl_h5_path: {hecvl_h5_path}")
    
    peskavlp_h5_path = os.path.join(OUTPUT_DIR, "peskavlp_fbk_vis_embs.h5")
    logger.info(f"peskavlp_h5_path: {peskavlp_h5_path}")
    print(f"peskavlp_h5_path: {peskavlp_h5_path}")
    
    videomae_urology_h5_path = os.path.join(OUTPUT_DIR, "videomae_urology_fbk_vis_embs.h5")
    logger.info(f"videomae_urology_h5_path: {videomae_urology_h5_path}")
    print(f"videomae_urology_h5_path: {videomae_urology_h5_path}")
    
    videomae_cholect45_h5_path = os.path.join(OUTPUT_DIR, "videomae_cholect45_fbk_vis_embs.h5")
    logger.info(f"videomae_cholect45_h5_path: {videomae_cholect45_h5_path}")
    print(f"videomae_cholect45_h5_path: {videomae_cholect45_h5_path}")
    
    
if __name__ == "__main__":
    main()
