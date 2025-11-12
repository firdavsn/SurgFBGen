from mmengine.config import Config
from typing import Dict, Any, List, Optional, Union
import os
from torchvision.transforms.transforms import Compose
from transformers import VideoMAEConfig, VideoMAEModel
import torch
import subprocess
import zipfile

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', 'surgical_VL_models')

CONFIG_PATHS = {
    'SurgVLP': os.path.join(CONFIGS_DIR, 'config_surgvlp.py'),
    'HecVL': os.path.join(CONFIGS_DIR, 'config_hecvl.py'),
    'PeskaVLP': os.path.join(CONFIGS_DIR, 'config_peskavlp.py'),
    'VideoMAE': os.path.join(CONFIGS_DIR, 'config_videomae.py'),
}

def surgvlp_load(
    model_config, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", download_root: str = None, pretrain: str=None, strict_load_state_dict: bool=True
):
    import torchvision.transforms as transforms
    from surgvlp.codes.models import build_algorithm
    
    _INPUT_RES = {
        "SurgVLP": 224,
        "HecVL": 224,
        "PeskaVLP": 224
    } 
    
    _MODELS = {
        "SurgVLP": "https://seafile.unistra.fr/f/93757ace1bfc47248e1e/?dl=1",
        "HecVL": "https://seafile.unistra.fr/f/3b9b9207068a4b03bc2a/?dl=1",
        "PeskaVLP": "https://seafile.unistra.fr/f/65a2b1bf113e428280d0/?dl=1",
    } 
        
    def _convert_image_to_rgb(image):
        return image.convert("RGB")
    
    def _transform(n_px):
        return transforms.Compose([
            transforms.Resize((360, 640)),
            transforms.CenterCrop(n_px),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
    def _download(models: Dict[str, str], key: str, root: str) -> str:
        url = models[key]
        os.makedirs(root, exist_ok=True)
        filename = key + '.zip'
        
        download_target = os.path.join(root, filename)
        
        if os.path.exists(download_target) and not os.path.isfile(download_target):
            raise RuntimeError(f"{download_target} exists and is not a regular file")
        
        if os.path.isfile(download_target):
            if zipfile.is_zipfile(download_target):
                with zipfile.ZipFile(download_target, 'r') as zip_ref:
                    zip_ref.extractall(root)
            return download_target.replace('.zip', '.pth')
            
        # Using wget to download the file with --content-disposition
        command = ['wget', '--content-disposition', '-P', root, url]
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download file: {result.stderr}")
        
        # Check if the downloaded file is a zip file and unzip it
        if zipfile.is_zipfile(download_target):
            with zipfile.ZipFile(download_target, 'r') as zip_ref:
                zip_ref.extractall(root)
        
        return download_target.replace('.zip', '.pth')
    
    def load(model_config, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", download_root: str = None, pretrain: str=None, strict_load_state_dict: bool=True):
        model_name = model_config['type']
        if pretrain is not None:
            model_path = pretrain
        else:
            model_path = _download(_MODELS, model_name, download_root or os.path.expanduser("~/.cache/surgvlp"))

        input_size = _INPUT_RES[model_name]

        model = build_algorithm(model_config).to(device)
        model.load_state_dict(torch.load(model_path), strict=strict_load_state_dict)
        model = model.eval()

        return model, _transform(input_size)
    return load(model_config, device, download_root, pretrain, strict_load_state_dict)

def load_VideoMAE_model(
    device: str = "cuda"
):
    configs = Config.fromfile(CONFIG_PATHS['VideoMAE'], lazy_import=False)['config']
    model = VideoMAEModel(configs.model_config).to(device)
    preprocess = configs.preprocess
    return model, preprocess