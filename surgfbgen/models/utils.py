from mmengine.config import Config
from typing import Dict, Any, List, Optional, Union
import os
from torchvision.transforms.transforms import Compose
from transformers import VideoMAEConfig, VideoMAEModel

import surgvlp
from surgvlp.codes.models.algorithms.SurgVLP import SurgVLP
from surgvlp.codes.models.algorithms.HecVL import HecVL
from surgvlp.codes.models.algorithms.PeskaVLP import PeskaVLP

import surgfbgen.config.environment

CONFIGS_DIR = os.path.join(os.environ['REPO_DIRECTORY'], 'config', 'surgical_VL_models')

CONFIG_PATHS = {
    'SurgVLP': os.path.join(CONFIGS_DIR, 'config_surgvlp.py'),
    'HecVL': os.path.join(CONFIGS_DIR, 'config_hecvl.py'),
    'PeskaVLP': os.path.join(CONFIGS_DIR, 'config_peskavlp.py'),
    'VideoMAE': os.path.join(CONFIGS_DIR, 'config_videomae.py'),
}

def load_surgical_VL_model(
    model_name: str, 
    device: str = "cuda"
) -> tuple[Union[SurgVLP, HecVL, PeskaVLP], Compose, callable]:
    configs = Config.fromfile(CONFIG_PATHS[model_name])['config']
    model, img_preprocess = surgvlp.load(configs.model_config, device=device)
    tokenizer = surgvlp.tokenize
    return model, img_preprocess, tokenizer

def load_VideoMAE_model(
    device: str = "cuda"
):
    configs = Config.fromfile(CONFIG_PATHS['VideoMAE'], lazy_import=False)['config']
    model = VideoMAEModel(configs.model_config).to(device)
    preprocess = configs.preprocess
    return model, preprocess