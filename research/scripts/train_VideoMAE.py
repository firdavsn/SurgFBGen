import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import VideoMAEConfig, VideoMAEForPreTraining, VideoMAEModel, set_seed, get_linear_schedule_with_warmup
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from torchmetrics import AveragePrecision
import pandas as pd
import math
from tqdm import tqdm
import random
from datetime import datetime
import json
import logging

# Define constants and hyperparameters
SEED = 42
DATA_DIR = '/home/firdavs/CholecT45'  # Replace with actual path to CholecT45 dataset
VIDEO_IDS_TRAIN = [f'VID{i:02d}' for i in [1, 2, 4, 5, 6, 
                                           8, 10, 12, 13, 14, 
                                           15, 18, 22, 23, 25, 
                                           26, 27, 29, 31, 32, 
                                           35, 36, 40, 42, 43, 
                                           47, 48, 49, 50, 51]]  # video 1 to 30
# VIDEO_IDS_TRAIN = [f'VID{i:02d}' for i in [1]]  # video 1 to 30
VIDEO_IDS_VAL = [f'VID{i:02d}' for i in [52, 56, 57, 60, 62]]   # video 31 to 35
# VIDEO_IDS_VAL = [f'VID{i:02d}' for i in [52]]   # video 31 to 35
VIDEO_IDS_TEST = [f'VID{i:02d}' for i in [65, 66, 68, 70, 73, 
                                          74, 75, 78, 79, 80]]  # video 36 to 45
# VIDEO_IDS_TEST = [f'VID{i:02d}' for i in [65]]  # video 36 to 45

BACKBONE_CONFIG = VideoMAEConfig()
SEQUENCE_LENGTH = 16            # Number of frames per sequence
STRIDE_PRETRAIN = 8             # Stride for sampling sequences in pretraining
STRIDE_FINETUNE = 8             # Stride for fine-tuning
STRIDE_EVAL = 1                 # Stride for evaluation (dense sampling)
BATCH_SIZE = 12                 # Batch size (adjust based on GPU memory)
NUM_EPOCHS_PRETRAIN = 50        # Number of pretraining epochs
NUM_EPOCHS_FINETUNE = 10        # Number of fine-tuning epochs
LEARNING_RATE = 5e-5            # Learning rate for optimization
SCHEDULER_WARMUP_RATIO = 0.1    # Warmup ratio for scheduler
MASK_RATIO = 0.75               # Mask ratio for VideoMAE
FREEZE_BACKBONE_LAYERS = False  # Whether to freeze the backbone layers

RUN_NAME = f'VideoMAE-finetune-add_fc_layers_epoch5-CholecT45-seed={SEED}-dt={datetime.now().strftime("%Y_%m_%d.%H_%M_%S")}'
print(f"RUN_NAME: {RUN_NAME}")
CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', RUN_NAME)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
ALL_CONSTANTS = {
    'SEED': SEED,
    "BACKBONE_CONFIG": BACKBONE_CONFIG.to_dict(),
    'DATA_DIR': DATA_DIR,
    'VIDEO_IDS_TRAIN': VIDEO_IDS_TRAIN,
    'VIDEO_IDS_VAL': VIDEO_IDS_VAL,
    'VIDEO_IDS_TEST': VIDEO_IDS_TEST,
    'SEQUENCE_LENGTH': SEQUENCE_LENGTH,
    'STRIDE_PRETRAIN': STRIDE_PRETRAIN,
    'STRIDE_FINETUNE': STRIDE_FINETUNE,
    'STRIDE_EVAL': STRIDE_EVAL,
    'BATCH_SIZE': BATCH_SIZE,
    'NUM_EPOCHS_PRETRAIN': NUM_EPOCHS_PRETRAIN,
    'NUM_EPOCHS_FINETUNE': NUM_EPOCHS_FINETUNE,
    'LEARNING_RATE': LEARNING_RATE,
    'SCHEDULER_WARMUP_RATIO': SCHEDULER_WARMUP_RATIO,
    'MASK_RATIO': MASK_RATIO,
    'RUN_NAME': RUN_NAME,
    'CHECKPOINTS_DIR': CHECKPOINTS_DIR,
}
for key, value in ALL_CONSTANTS.items():
    print(f"{key}: {value}")
with open(os.path.join(CHECKPOINTS_DIR, 'constants.json'), 'w') as f:
    json.dump(ALL_CONSTANTS, f, indent=4)

# Define image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize frames to 224x224 as expected by VideoMAE
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set seed: {seed}")

set_all_seeds(SEED)

# Dataset class for pretraining (loads video frames only)
class CholecT45PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, video_ids, sequence_length, stride, transform=None):
        self.data_dir = data_dir
        self.video_ids = video_ids
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        self.frame_indices = {}  # Store frame indices for each video
        self.samples = []
        
        # Load frame indices from triplet annotation files
        for vid in video_ids:
            annot_file = os.path.join(data_dir, 'triplet', f'{vid}.txt')
            with open(annot_file, 'r') as f:
                data = pd.read_csv(f, header=None)
            frame_indices = data[0].tolist()
            self.frame_indices[vid] = frame_indices
            # Generate samples using actual frame indices
            for start_idx in range(0, len(frame_indices) - sequence_length + 1, stride):
                sequence_frame_ids = frame_indices[start_idx:start_idx + sequence_length]
                self.samples.append((vid, sequence_frame_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid, frame_ids = self.samples[idx]
        frames = []
        for frame_id in frame_ids:
            frame_path = os.path.join(self.data_dir, 'data', vid, f'{frame_id:06d}.png')
            frame = Image.open(frame_path).convert('RGB')
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        return torch.stack(frames)

# Dataset class for fine-tuning (loads frames and labels)
class CholecT45FinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, video_ids, sequence_length, stride, transform=None):
        self.data_dir = data_dir
        self.video_ids = video_ids
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        self.frame_indices = {}
        self.labels = {'inst': {}, 'verb': {}, 'target': {}}
        
        # Load frame indices and labels
        for vid in video_ids:
            # Load frame indices from triplet file
            annot_file = os.path.join(data_dir, 'triplet', f'{vid}.txt')
            with open(annot_file, 'r') as f:
                data = pd.read_csv(f, header=None)
            frame_indices = data[0].tolist()
            self.frame_indices[vid] = frame_indices
            # Load labels for each category
            for cat, num_classes in [('inst', 6), ('verb', 10), ('target', 15)]:
                annot_file = os.path.join(data_dir, cat + 'rument' if cat == 'inst' else cat, f'{vid}.txt')
                data = pd.read_csv(annot_file, header=None)
                self.labels[cat][vid] = np.array(data.values.tolist())[:, 1:]
        
        # Generate samples
        self.samples = []
        for vid in video_ids:
            indices = self.frame_indices[vid]
            for start_idx in range(0, len(indices) - sequence_length + 1, stride):
                sequence_frame_ids = indices[start_idx:start_idx + sequence_length]
                self.samples.append((vid, sequence_frame_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid, frame_ids = self.samples[idx]
        frames = []
        for frame_id in frame_ids:
            frame_path = os.path.join(self.data_dir, 'data', vid, f'{frame_id:06d}.png')
            frame = Image.open(frame_path).convert('RGB')
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        sequence = torch.stack(frames)
        # Get labels for the last frame
        annot_idx = frame_ids[-1]
        inst_labels = torch.tensor(self.labels['inst'][vid][annot_idx], dtype=torch.float)
        verb_labels = torch.tensor(self.labels['verb'][vid][annot_idx], dtype=torch.float)
        target_labels = torch.tensor(self.labels['target'][vid][annot_idx], dtype=torch.float)
        return sequence, inst_labels, verb_labels, target_labels

# Custom model for multi-label classification
class VideoMAEForMultiLabelClassification(nn.Module):
    def __init__(self, config, freeze_backbone_layers=False):
        super().__init__()
        self.videomae = VideoMAEModel(config)  # Encoder only
        if freeze_backbone_layers:
            for param in self.videomae.parameters():
                param.requires_grad = False
        
        self.net = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        # self.inst_head = nn.Linear(config.hidden_size, 6)   # 6 instrument classes
        # self.verb_head = nn.Linear(config.hidden_size, 10)  # 10 verb classes
        # self.target_head = nn.Linear(config.hidden_size, 15)  # 15 target classes
        
        self.inst_head = nn.Linear(128, 6)   # 6 instrument classes
        self.verb_head = nn.Linear(128, 10)  # 10 verb classes
        self.target_head = nn.Linear(128, 15)  # 15 target classes
        
        self.config = config

    def forward(self, pixel_values):
        outputs = self.videomae(pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token
        x = self.net(cls_token)
        
        # inst_logits = self.inst_head(cls_token)
        # verb_logits = self.verb_head(cls_token)
        # target_logits = self.target_head(cls_token)
        
        inst_logits = self.inst_head(x)
        verb_logits = self.verb_head(x)
        target_logits = self.target_head(x)
        
        return inst_logits, verb_logits, target_logits

# Model setup functions
def setup_pretrain_model():
    """Initialize VideoMAE model for pretraining."""
    model = VideoMAEForPreTraining(BACKBONE_CONFIG)
    return model

def setup_finetune_model(pretrained_model):
    """Initialize fine-tuning model with pretrained encoder weights."""
    # Get pretrained model config
    config = pretrained_model.config
    model = VideoMAEForMultiLabelClassification(config, freeze_backbone_layers=FREEZE_BACKBONE_LAYERS)
    model.videomae.load_state_dict(pretrained_model.videomae.state_dict())  # Load encoder weights
    return model

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

# Training functions
def pretrain(model, train_loader, optimizer, scheduler, device, logger):
    """Pretrain the VideoMAE model using masked autoencoding."""
    model.train()
    for epoch in range(NUM_EPOCHS_PRETRAIN):
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Pre-training Epoch {epoch+1}/{NUM_EPOCHS_PRETRAIN}'):
            pixel_values = batch.to(device)
                
            # Get configuration values
            patch_size = model.config.patch_size  # e.g., 16 (integer)
            tubelet_size = model.config.tubelet_size  # e.g., 2 (integer)
            image_size = model.config.image_size  # e.g., (224, 224) or 224
            num_frames = pixel_values.size(1)  # e.g., 16
            
            # Handle image_size (could be tuple or integer)
            if isinstance(image_size, int):
                height = width = image_size
            else:
                height, width = image_size
            
            # Calculate number of spatial patches per frame
            num_patches_per_frame = (height // patch_size) * (width // patch_size)  # e.g., 14*14=196
            # Calculate number of temporal patches
            num_temporal_patches = num_frames // tubelet_size  # e.g., 16//2=8
            # Total number of spatio-temporal patches
            total_patches = num_temporal_patches * num_patches_per_frame  # e.g., 8*196=1568
            
            # Generate random boolean mask for patches
            batch_size = pixel_values.size(0)
            bool_masked_pos = np.ones(total_patches)
            mask_num = math.ceil(total_patches * MASK_RATIO)
            mask = np.random.choice(total_patches, mask_num, replace=False)
            bool_masked_pos[mask] = 0

            # Torch and bool cast, extra dimension added for concatenation
            bool_masked_pos = torch.as_tensor(bool_masked_pos).bool().unsqueeze(0)
            bool_masked_pos = torch.cat([bool_masked_pos for _ in range(batch_size)])
            
            # Call the model with pixel_values and bool_masked_pos
            outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        logger.info(f"Pretraining Epoch {epoch+1}/{NUM_EPOCHS_PRETRAIN}, Train Loss: {train_loss:.4f}")
        print(f"Pretraining Epoch {epoch+1}/{NUM_EPOCHS_PRETRAIN}, Train Loss: {train_loss:.4f}")
        
        # Save checkpoint
        checkpoint_dir = os.path.join(CHECKPOINTS_DIR, 'pretrain')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'model_config': model.config,
        }, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        print(f"Checkpoint saved to {checkpoint_path}")


def finetune(model, train_loader, val_loader, optimizer, scheduler, device, logger):
    """Fine-tune the model for multi-label classification."""
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(NUM_EPOCHS_FINETUNE):
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Fine-tuning Epoch {epoch+1}/{NUM_EPOCHS_FINETUNE} - Train'):
            pixel_values, inst_labels, verb_labels, target_labels = batch
            pixel_values = pixel_values.to(device)
            inst_labels = inst_labels.to(device)
            verb_labels = verb_labels.to(device)
            target_labels = target_labels.to(device)
            inst_logits, verb_logits, target_logits = model(pixel_values)
            loss_inst = loss_fn(inst_logits, inst_labels)
            loss_verb = loss_fn(verb_logits, verb_labels)
            loss_target = loss_fn(target_logits, target_labels)
            loss = loss_inst + loss_verb + loss_target  # Sum losses
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        logger.info(f"Fine-tuning Epoch {epoch+1}/{NUM_EPOCHS_FINETUNE}, Train Loss: {train_loss:.4f}")
        print(f"Fine-tuning Epoch {epoch+1}/{NUM_EPOCHS_FINETUNE}, Train Loss: {train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Fine-tuning Epoch {epoch+1}/{NUM_EPOCHS_FINETUNE} - Val'):
                pixel_values, inst_labels, verb_labels, target_labels = batch
                pixel_values = pixel_values.to(device)
                inst_labels = inst_labels.to(device)
                verb_labels = verb_labels.to(device)
                target_labels = target_labels.to(device)
                inst_logits, verb_logits, target_logits = model(pixel_values)
                loss_inst = loss_fn(inst_logits, inst_labels)
                loss_verb = loss_fn(verb_logits, verb_labels)
                loss_target = loss_fn(target_logits, target_labels)
                val_loss += (loss_inst + loss_verb + loss_target).item()
        val_loss /= len(val_loader)
        logger.info(f"Fine-tuning Epoch {epoch+1}/{NUM_EPOCHS_FINETUNE}, Val Loss: {val_loss:.4f}")
        print(f"Fine-tuning Epoch {epoch+1}/{NUM_EPOCHS_FINETUNE}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        checkpoint_dir = os.path.join(CHECKPOINTS_DIR, 'finetune')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'model_config': model.config,
        }, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        print(f"Checkpoint saved to {checkpoint_path}")
        
        model.train()

# Evaluation function
def evaluate(model, test_loader, device):
    """Evaluate the model using mean average precision (mAP)."""
    model.eval()
    ap_inst = AveragePrecision(num_labels=6, average='macro', task='multilabel')
    ap_verb = AveragePrecision(num_labels=10, average='macro', task='multilabel')
    ap_target = AveragePrecision(num_labels=15, average='macro', task='multilabel')
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f'Evaluation'):
            pixel_values, inst_labels, verb_labels, target_labels = batch
            pixel_values = pixel_values.to(device)
            inst_labels = inst_labels.to(device)
            verb_labels = verb_labels.to(device)
            target_labels = target_labels.to(device)
            inst_logits, verb_logits, target_logits = model(pixel_values)
            inst_preds = torch.nn.functional.sigmoid(inst_logits)  # Convert logits to probabilities
            verb_preds = torch.nn.functional.sigmoid(verb_logits)
            target_preds = torch.nn.functional.sigmoid(target_logits)
            
            ap_inst.update(inst_preds, inst_labels.int())
            ap_verb.update(verb_preds, verb_labels.int())
            ap_target.update(target_preds, target_labels.int())

    inst_map = float(ap_inst.compute())
    verb_map = float(ap_verb.compute())
    target_map = float(ap_target.compute())
    print(f"Instrument mAP: {inst_map:.4f}")
    print(f"Verb mAP: {verb_map:.4f}")
    print(f"Target mAP: {target_map:.4f}")
    
    return {
        'inst_map': inst_map,
        'verb_map': verb_map,
        'target_map': target_map
    }

def load_model(model_path):
    with torch.serialization.safe_globals([VideoMAEConfig]):
        model = torch.load(model_path)
    return model


# Main execution function
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize logger
    log_file = os.path.join(CHECKPOINTS_DIR, 'logging.log')
    logger = setup_logger(log_file)

    # Pretraining phase
    logger.info("Starting pretraining...")
    print("Starting pretraining...")
    pretrain_dataset = CholecT45PretrainDataset(
        DATA_DIR, VIDEO_IDS_TRAIN, SEQUENCE_LENGTH, STRIDE_PRETRAIN, transform
    )
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True)
    pretrain_model = setup_pretrain_model().to(device)
    optimizer = torch.optim.AdamW(pretrain_model.parameters(), lr=LEARNING_RATE)
    total_steps_pretrain = len(pretrain_loader) * NUM_EPOCHS_PRETRAIN
    warmup_steps_pretrain = int(SCHEDULER_WARMUP_RATIO * total_steps_pretrain)
    scheduler_pretrain = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps_pretrain,
        num_training_steps=total_steps_pretrain
    )
    # pretrain(pretrain_model, pretrain_loader, optimizer, scheduler_pretrain, device, logger)

    # Load pretrained model
    pretrained_model_path = '/home/firdavs/surgery/surgical_fb_generation/SurgicalFeedbackGeneration/checkpoints/VideoMAE-CholecT45-seed=42-dt=2025_03_08.20_30_38/pretrain/epoch_50.pth'
    # pretrained_model_path = None
    if pretrained_model_path is not None:
        logger.info(f"Loading pretrained model from {pretrained_model_path}...")
        print(f"Loading pretrained model from {pretrained_model_path}...")
        pretrained_model = load_model(pretrained_model_path)
        pretrain_model.load_state_dict(pretrained_model['model_state_dict'])
    else:
        logger.info("No pretrained model provided.")
        print("No pretrained model provided.")
        
    # Fine-tuning phase
    logger.info("Starting fine-tuning...")
    print("Starting fine-tuning...")
    finetune_dataset_train = CholecT45FinetuneDataset(
        DATA_DIR, VIDEO_IDS_TRAIN, SEQUENCE_LENGTH, STRIDE_FINETUNE, transform
    )
    finetune_loader_train = DataLoader(finetune_dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    finetune_dataset_val = CholecT45FinetuneDataset(
        DATA_DIR, VIDEO_IDS_VAL, SEQUENCE_LENGTH, STRIDE_EVAL, transform
    )
    finetune_loader_val = DataLoader(finetune_dataset_val, batch_size=BATCH_SIZE, shuffle=False)
    finetune_model = setup_finetune_model(pretrain_model).to(device)
    optimizer = torch.optim.AdamW(finetune_model.parameters(), lr=LEARNING_RATE)
    total_steps_finetune = len(finetune_loader_train) * NUM_EPOCHS_FINETUNE
    warmup_steps_finetune = int(SCHEDULER_WARMUP_RATIO * total_steps_finetune)
    scheduler_finetune = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps_finetune,
        num_training_steps=total_steps_finetune
    )
    # finetune(finetune_model, finetune_loader_train, finetune_loader_val, optimizer, scheduler_finetune, device, logger)

    # Load finetuned model
    finetuned_model_path = '/home/firdavs/surgery/surgical_fb_generation/SurgicalFeedbackGeneration/checkpoints/VideoMAE-finetune-add_fc_layers-CholecT45-seed=42-dt=2025_03_10.20_30_15/finetune/epoch_5.pth'
    # finetuned_model_path = None
    if finetuned_model_path is not None:
        logger.info(f"Loading finetuned model from {finetuned_model_path}...")
        print(f"Loading finetuned model from {finetuned_model_path}...")
        finetuned_model = load_model(finetuned_model_path)
        finetune_model.load_state_dict(finetuned_model['model_state_dict'])
    else:
        logger.info("No finetuned model provided.")
        print("No finetuned model provided.")
    
    # Evaluation phase
    logger.info("Starting evaluation...")
    print("Starting evaluation...")
    test_dataset = CholecT45FinetuneDataset(
        DATA_DIR, VIDEO_IDS_TEST, SEQUENCE_LENGTH, STRIDE_EVAL, transform
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    results = evaluate(finetune_model, test_loader, device)
    results_path = os.path.join(CHECKPOINTS_DIR, 'results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info("Evaluation complete. Results saved to %s", results_path)
    print("Evaluation complete. Results saved to %s", results_path)

if __name__ == '__main__':
    main()
