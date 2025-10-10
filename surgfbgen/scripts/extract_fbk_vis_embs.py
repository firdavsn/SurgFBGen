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
from transformers import AutoImageProcessor, VideoMAEModel, VideoMAEForPreTraining, AutoVideoProcessor, AutoModel
from pytorchvideo.transforms import UniformTemporalSubsample

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SurgVLP_path = '/home/firdavs/surgery/surgical_fb_generation/SurgVLP'
CLIPS_DATA_DIR = "/home/firdavs/surgery/clips_with_wiggle/fb_clips_wiggle"
CLIP_FILE_TYPE = 'avi'
LOGGING_PATH = f'/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/surgfbgen/logs/extract_fbk_vis_embs-dt=({datetime.now().strftime("%Y-%m-%d-%H-%M")}).log'
OUTPUT_DIR = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/outputs/embeddings/vision'

import sys
sys.path.append(SurgVLP_path)
import surgvlp

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

def load_clip_frames(path: str, logger: Logger = None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        if logger:
            logger.error(f"Error opening video from {path}")
        else:
            print(f"Error opening video from {path}")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            if logger:
                logger.error(f"Error reading frame {frame_idx} from {path}")
            else:
                print(f"Error reading frame {frame_idx} from {path}")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames, total_frames

def process_frames_vlp(frames, processor, logger: Logger = None):
    processed_frames = []
    for frame in frames:
        processed_frame = processor(Image.fromarray(frame))
        processed_frames.append(processed_frame)
    return processed_frames

def process_frames_videomae(frames, processor, logger: Logger = None, num_sample_frames: int = 16):
    frames = torch.Tensor(np.array(frames).transpose(3, 0, 1, 2))
    subsampler = UniformTemporalSubsample(num_sample_frames)
    subsampled_frames = subsampler(frames)
    video_data_np = subsampled_frames.numpy().transpose(1, 0, 2, 3)

    # Preprocess video data
    frame_list = list(video_data_np)
    video_batch = [frame_list]
    inputs = processor.preprocess(video_batch, return_tensors="pt")
    return inputs

def get_vlp_embeddings(model, preprocessor, clips_data_dir: str, output_h5_path: str, device: str, logger: Logger = None, clip_file_type: str = 'avi', overwrite: bool = False):
    # Get all clip paths
    file2path = {f: os.path.join(clips_data_dir, f) for f in os.listdir(clips_data_dir) if f.endswith(clip_file_type)}
    files = list(file2path.keys())
    if logger:
        logger.info(f"Found {len(file2path)} clips")
    print(f"Found {len(file2path)} clips")
    
    # Check if output file exists and contains data
    if os.path.exists(output_h5_path):
        if overwrite:
            h5 = h5py.File(output_h5_path, 'w')
        else:
            h5 = h5py.File(output_h5_path, 'a')
    else:
        h5 = h5py.File(output_h5_path, 'a')
    
    # Extract embeddings
    save_every = 50
    if not overwrite:
        processed_files = list(h5.keys())
        remaining_files = [f for f in files if f not in processed_files]
        files = remaining_files
    i = 0
    iterator = tqdm(files) if logger else files
    for file in iterator:
        if logger:
            logger.info(f"Processing {i+1}/{len(file2path)}: {file}")
        else:
            print(f"Processing {i+1}/{len(file2path)}: {file}")
        frames, total_frames = load_clip_frames(file2path[file], logger)
        i += 1
        
        if i % save_every == 0:
            h5.flush()
        
        if frames is None:
            if logger:
                logger.error(f"No frames loaded for {file}")
            else:
                print(f"No frames loaded for {file}")
            continue
        
        frames = process_frames_vlp(frames, preprocessor, logger)
        
        frames_tensor = torch.stack(frames).to(device)
        with torch.no_grad():
            img_emb = model(frames_tensor, None, mode='video')['img_emb'].cpu().numpy()  # (total_frames, 768)
        h5.create_dataset(file, data=img_emb)

def get_videomae_embeddings(model, preprocessor, clips_data_dir: str, output_h5_path: str, device: str, logger: Logger = None, clip_file_type: str = 'avi'):
    # Get all clip paths
    file2path = {f: os.path.join(clips_data_dir, f) for f in os.listdir(clips_data_dir) if f.endswith(clip_file_type)}
    files = list(file2path.keys())
    if logger:
        logger.info(f"Found {len(file2path)} clips")
    print(f"Found {len(file2path)} clips")
    
    # Extract embeddings
    with h5py.File(output_h5_path, 'w') as h5:
        i = 0
        iterator = tqdm(files) if logger else files
        for file in iterator:
            if logger:
                logger.info(f"Processing {i+1}/{len(file2path)}: {file}")
            else:
                print(f"Processing {i+1}/{len(file2path)}: {file}")
            frames, total_frames = load_clip_frames(file2path[file], logger)
            i += 1
            
            if frames is None:
                if logger:
                    logger.error(f"No frames loaded for {file}")
                else:
                    print(f"No frames loaded for {file}")
                continue
            
            inputs = process_frames_videomae(frames, preprocessor, logger, num_sample_frames=16)
            
            with torch.no_grad():
                img_emb = model(inputs.pixel_values).last_hidden_state.squeeze().cpu().numpy()  # (16 * num_patches, 768)
                # img_emb = img_emb.mean(axis=1)
                print(img_emb.shape)
            h5.create_dataset(file, data=img_emb)    

def surgvlp_embeddings(clips_data_dir: str, output_h5_path: str, device: str, logger: Logger = None, clip_file_type: str = 'avi', overwrite: bool = False):
    print("Extracting SurgVLP embeddings...")
    if logger:
        logger.info("Extracting PeskaVLP embeddings...")
        
    # Load model and preprocessor
    configs = Config.fromfile(os.path.join(SurgVLP_path, 'tests', 'config_surgvlp.py'), lazy_import=False)['config']
    model, preprocessor = surgvlp.load(configs.model_config, device=device, strict_load_state_dict=False)
    
    # Extract embeddings
    get_vlp_embeddings(model, preprocessor, clips_data_dir, output_h5_path, device, logger, clip_file_type, overwrite=overwrite)


def hecvl_embeddings(clips_data_dir: str, output_h5_path: str, device: str, logger: Logger = None, clip_file_type: str = 'avi'):
    print("Extracting HecVL embeddings...")
    if logger:
        logger.info("Extracting PeskaVLP embeddings...")
        
    # Load model and preprocessor
    configs = Config.fromfile(os.path.join(SurgVLP_path, 'tests', 'config_hecvl.py'), lazy_import=False)['config']
    model, preprocessor = surgvlp.load(configs.model_config, device=device, strict_load_state_dict=False)
    
    # Extract embeddings
    get_vlp_embeddings(model, preprocessor, clips_data_dir, output_h5_path, device, logger, clip_file_type)
    

def peskavlp_embeddings(clips_data_dir: str, output_h5_path: str, device: str, logger: Logger = None, clip_file_type: str = 'avi'):
    print("Extracting PeskaVLP embeddings...")
    if logger:
        logger.info("Extracting PeskaVLP embeddings...")
        
    # Load model and preprocessor
    configs = Config.fromfile(os.path.join(SurgVLP_path, 'tests', 'config_peskavlp.py'), lazy_import=False)['config']
    model, preprocessor = surgvlp.load(configs.model_config, device=device, strict_load_state_dict=False)
    
    # Extract embeddings
    get_vlp_embeddings(model, preprocessor, clips_data_dir, output_h5_path, device, logger, clip_file_type)
    

def videomae_urology_embeddings(clips_data_dir: str, output_h5_path: str, device: str, logger: Logger = None, clip_file_type: str = 'avi'):
    print("Extracting VideoMAE-Urology embeddings...")
    if logger:
        logger.info("Extracting VideoMAE-Urology embeddings...")
        
    # Load model and preprocessor
    model = VideoMAEModel.from_pretrained("/home/firdavs/surgery/surgical_fb_generation/SurgicalFeedbackGeneration/checkpoints/VideoMAE-urology-pretrain-from_Arushi/")
    preprocessor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    
    # Extract embeddings
    get_videomae_embeddings(model, preprocessor, clips_data_dir, output_h5_path, device, logger, clip_file_type)

def videomae_cholect45_embeddings(clips_data_dir: str, output_h5_path: str, device: str, logger: Logger = None, clip_file_type: str = 'avi'):
    print("Extracting VideoMAE-CholecT45 embeddings...")
    if logger:
        logger.info("Extracting VideoMAE-CholecT45 embeddings...")
        
    # Load model and preprocessor
    model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")
    with open('/home/firdavs/surgery/surgical_fb_generation/SurgicalFeedbackGeneration/checkpoints/VideoMAE-CholecT45-seed=42-dt=2025_03_08.20_30_38/pretrain/epoch_50.pth', 'rb') as f:
        epoch = torch.load(f, weights_only=False)
        state_dict = epoch['model_state_dict']
        model_pretraining = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")
        model_pretraining.load_state_dict(state_dict, strict=False)
    model = model.videomae
    preprocessor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    
    # Extract embeddings
    get_videomae_embeddings(model, preprocessor, clips_data_dir, output_h5_path, device, logger, clip_file_type)
    
def videomae_base_embeddings(clips_data_dir: str, output_h5_path: str, device: str, logger: Logger = None, clip_file_type: str = 'avi'):
    print("Extracting VideoMAE-Base embeddings...")
    if logger:
        logger.info("Extracting VideoMAE-Base embeddings...")
        
    # Load model and preprocessor
    print("Loading VideoMAE-Base model...")
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    preprocessor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    print("VideoMAE-Base model loaded")
    
    # Extract embeddings
    print("Extracting VideoMAE-Base embeddings...")
    get_videomae_embeddings(model, preprocessor, clips_data_dir, output_h5_path, device, logger, clip_file_type)
        

import decord
def _pe_preprocess_video(video_path, num_frames=8, transform=None, return_first_frame_for_demo=True):
    """
    Uniformly samples a specified number of frames from a video and preprocesses them.
    Parameters:
    - video_path: str, path to the video file.
    - num_frames: int, number of frames to sample. Defaults to 8.
    - transform: torchvision.transforms, a transform function to preprocess frames.
    Returns:
    - Video Tensor: a tensor of shape (num_frames, 3, H, W) where H and W are the height and width of the frames.
    """
    # Load the video
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    # Uniformly sample frame indices
    frame_indices = [int(i * (total_frames / num_frames)) for i in range(num_frames)]
    frames = vr.get_batch(frame_indices).asnumpy()
    # Preprocess frames
    preprocessed_frames = [transform(Image.fromarray(frame)) for frame in frames]

    first_frame = None
    if return_first_frame_for_demo:
        first_frame = frames[0]
    return torch.stack(preprocessed_frames, dim=0), first_frame

def get_pe_embeddings(model, preprocessor, clips_data_dir: str, output_h5_path: str, device: str, logger: Logger = None, clip_file_type: str = 'avi', overwrite: bool = False):
    # Get all clip paths
    file2path = {f: os.path.join(clips_data_dir, f) for f in os.listdir(clips_data_dir) if f.endswith(clip_file_type)}
    files = list(file2path.keys())
    if logger:
        logger.info(f"Found {len(file2path)} clips")
    print(f"Found {len(file2path)} clips")
    
    # Check if output file exists and contains data
    if os.path.exists(output_h5_path):
        if overwrite:
            h5 = h5py.File(output_h5_path, 'w')
        else:
            h5 = h5py.File(output_h5_path, 'a')
    else:
        h5 = h5py.File(output_h5_path, 'a')
    
    # Extract embeddings
    save_every = 50
    if not overwrite:
        processed_files = list(h5.keys())
        remaining_files = [f for f in files if f not in processed_files]
        files = remaining_files

    i = 0
    iterator = tqdm(files) if logger else files
    for file in iterator:
        if logger:
            logger.info(f"Processing {i+1}/{len(file2path)}: {file}")
        else:
            print(f"Processing {i+1}/{len(file2path)}: {file}")
        frames, total_frames = load_clip_frames(file2path[file], logger)
        i += 1
        
        if i % save_every == 0:
            h5.flush()
        
        if frames is None:
            if logger:
                logger.error(f"No frames loaded for {file}")
            else:
                print(f"No frames loaded for {file}")
            continue
        
        frames = process_frames_vlp(frames, preprocessor, logger)
        
        frames_tensor = torch.stack(frames).to(device).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            img_emb = model.encode_video(frames_tensor)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)  # Normalize embeddings
            img_emb = img_emb.cpu().numpy()  # Convert to numpy array
        h5.create_dataset(file, data=img_emb)
    

def pe_embeddings(clips_data_dir: str, output_h5_path: str, device: str, logger: Logger = None, clip_file_type: str = 'avi', pixel_size: int = 224, overwrite: bool = False):
    # PE: Perception Encoder from Meta
    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as transforms
    print("CLIP configs:", pe.CLIP.available_configs())
    
    print(f"Extracting PE{pixel_size} embeddings...")
    if logger:
        logger.info(f"Extracting PE{pixel_size} embeddings...")
    
    # Load model and preprocessor
    name = None
    if pixel_size == 224:
        name = 'PE-Core-B16-224'
    elif pixel_size == 336:
        name = 'PE-Core-L14-336'
    elif pixel_size == 448:
        name = 'PE-Core-G14-448'
    else:
        raise ValueError(f"Unsupported pixel size: {pixel_size}. Supported sizes are 224, 336, and 448.")
    model = pe.CLIP.from_config(name, pretrained=True)  # Downloads from HF
    if device == "cuda":
        model = model.cuda()
    preprocessor = transforms.get_image_transform(model.image_size)
    
    # Extract embeddings
    get_pe_embeddings(model, preprocessor, clips_data_dir, output_h5_path, device, logger, clip_file_type, overwrite=overwrite)

def process_frames_vjepa2(frames, processor, logger: Logger = None, num_sample_frames: int = 16):
    frames = torch.Tensor(np.array(frames).transpose(3, 0, 1, 2))
    subsampler = UniformTemporalSubsample(num_sample_frames)
    subsampled_frames = subsampler(frames)
    video_data_np = subsampled_frames.numpy().transpose(1, 0, 2, 3)

    # Preprocess video data
    frame_list = list(video_data_np)
    video_batch = [frame_list]
    inputs = processor.preprocess(video_batch, return_tensors="pt").to('cuda')
    return inputs

def get_vjepa2_embeddings(model, preprocessor, clips_data_dir: str, output_h5_path: str, device: str, logger: Logger = None, clip_file_type: str = 'avi', overwrite: bool = False):
    # Get all clip paths
    file2path = {f: os.path.join(clips_data_dir, f) for f in os.listdir(clips_data_dir) if f.endswith(clip_file_type)}
    files = list(file2path.keys())
    if logger:
        logger.info(f"Found {len(file2path)} clips")
    print(f"Found {len(file2path)} clips")
    
    # Check if output file exists and contains data
    if os.path.exists(output_h5_path):
        if overwrite:
            h5 = h5py.File(output_h5_path, 'w')
        else:
            h5 = h5py.File(output_h5_path, 'a')
    else:
        h5 = h5py.File(output_h5_path, 'a')
    
    # Extract embeddings
    save_every = 50
    if not overwrite:
        processed_files = list(h5.keys())
        remaining_files = [f for f in files if f not in processed_files]
        files = remaining_files

        i = 0
        iterator = tqdm(files) if logger else files
        for file in iterator:
            if logger:
                logger.info(f"Processing {i+1}/{len(file2path)}: {file}")
            else:
                print(f"Processing {i+1}/{len(file2path)}: {file}")
            frames, total_frames = load_clip_frames(file2path[file], logger)
            i += 1
            
            if frames is None:
                if logger:
                    logger.error(f"No frames loaded for {file}")
                else:
                    print(f"No frames loaded for {file}")
                continue
            
            inputs = process_frames_vjepa2(frames, preprocessor, logger, num_sample_frames=16)
            
            with torch.no_grad():
                outputs = model(**inputs)
                encoder_outputs = outputs.last_hidden_state
                video_emb = encoder_outputs.mean(axis=1).cpu().numpy()
            h5.create_dataset(file, data=video_emb)    

def vjepa2_embeddings(clips_data_dir: str, output_h5_path: str, device: str, logger: Logger = None, clip_file_type: str = 'avi', overwrite: bool = False):
    print("Extracting VJepa2 embeddings...")
    if logger:
        logger.info("Extracting VJepa2 embeddings...")
        
    # Load model and preprocessor
    model = AutoModel.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256",
        device_map="auto",
        attn_implementation="sdpa"
    )
    preprocessor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
    model.to(device)

    # Extract embeddings
    get_vjepa2_embeddings(model, preprocessor, clips_data_dir, output_h5_path, device, logger, clip_file_type, overwrite=overwrite)


def main():
    logger = setup_logger(LOGGING_PATH)
    set_seed_all(SEED, logger)
    
    # surgvlp_h5_path = os.path.join(OUTPUT_DIR, "surgvlp_fbk_vis_embs.h5")
    # logger.info(f"surgvlp_h5_path: {surgvlp_h5_path}")
    # print(f"surgvlp_h5_path: {surgvlp_h5_path}")
    # surgvlp_embeddings(CLIPS_DATA_DIR, surgvlp_h5_path, DEVICE, logger, overwrite=False)
        
    # hecvl_h5_path = os.path.join(OUTPUT_DIR, "hecvl_fbk_vis_embs.h5")
    # logger.info(f"hecvl_h5_path: {hecvl_h5_path}")
    # print(f"hecvl_h5_path: {hecvl_h5_path}")
    # if not os.path.exists(hecvl_h5_path):
    #     hecvl_embeddings(CLIPS_DATA_DIR, hecvl_h5_path, DEVICE, logger)
    
    # peskavlp_h5_path = os.path.join(OUTPUT_DIR, "peskavlp_fbk_vis_embs.h5")
    # logger.info(f"peskavlp_h5_path: {peskavlp_h5_path}")
    # print(f"peskavlp_h5_path: {peskavlp_h5_path}")
    # if not os.path.exists(peskavlp_h5_path):
    #     peskavlp_embeddings(CLIPS_DATA_DIR, peskavlp_h5_path, DEVICE, logger)
    
    videomae_base_h5_path = os.path.join(OUTPUT_DIR, "videomae_base_fbk_vis_embs.h5")
    logger.info(f"videomae_base_h5_path: {videomae_base_h5_path}")
    print(f"videomae_base_h5_path: {videomae_base_h5_path}")
    videomae_base_embeddings(CLIPS_DATA_DIR, videomae_base_h5_path, DEVICE, logger)
    
    # videomae_urology_h5_path = os.path.join(OUTPUT_DIR, "videomae_urology_fbk_vis_embs.h5")
    # logger.info(f"videomae_urology_h5_path: {videomae_urology_h5_path}")
    # print(f"videomae_urology_h5_path: {videomae_urology_h5_path}")
    # if not os.path.exists(videomae_urology_h5_path):
    #     videomae_urology_embeddings(CLIPS_DATA_DIR, videomae_urology_h5_path, DEVICE, logger)
    
    # videomae_cholect45_h5_path = os.path.join(OUTPUT_DIR, "videomae_cholect45_fbk_vis_embs.h5")
    # logger.info(f"videomae_cholect45_h5_path: {videomae_cholect45_h5_path}")
    # print(f"videomae_cholect45_h5_path: {videomae_cholect45_h5_path}")
    # if not os.path.exists(videomae_cholect45_h5_path):
    #     videomae_cholect45_embeddings(CLIPS_DATA_DIR, videomae_cholect45_h5_path, DEVICE, logger)
    
    # pe224_h5_path = os.path.join(OUTPUT_DIR, "pe224_fbk_vis_embs.h5")
    # logger.info(f"pe224_h5_path: {pe224_h5_path}")
    # print(f"pe224_h5_path: {pe224_h5_path}")
    # pe_embeddings(CLIPS_DATA_DIR, pe224_h5_path, DEVICE, logger, clip_file_type=CLIP_FILE_TYPE, pixel_size=224, overwrite=False)
    
    # pe336_h5_path = os.path.join(OUTPUT_DIR, "pe336_fbk_vis_embs.h5")
    # logger.info(f"pe336_h5_path: {pe336_h5_path}")
    # print(f"pe336_h5_path: {pe336_h5_path}")
    # pe_embeddings(CLIPS_DATA_DIR, pe336_h5_path, DEVICE, logger, clip_file_type=CLIP_FILE_TYPE, pixel_size=336, overwrite=False)
        
    # pe448_h5_path = os.path.join(OUTPUT_DIR, "pe448_fbk_vis_embs.h5")
    # logger.info(f"pe448_h5_path: {pe448_h5_path}")
    # print(f"pe448_h5_path: {pe448_h5_path}")
    # pe_embeddings(CLIPS_DATA_DIR, pe448_h5_path, DEVICE, logger, clip_file_type=CLIP_FILE_TYPE, pixel_size=448, overwrite=False)
    
    # vjepa2_h5_path = os.path.join(OUTPUT_DIR, "vjepa2_fbk_vis_embs.h5")
    # logger.info(f"vjepa2_h5_path: {vjepa2_h5_path}")
    # print(f"vjepa2_h5_path: {vjepa2_h5_path}")
    # vjepa2_embeddings(CLIPS_DATA_DIR, vjepa2_h5_path, DEVICE, logger, clip_file_type=CLIP_FILE_TYPE, overwrite=False)
    
if __name__ == "__main__":
    main()
