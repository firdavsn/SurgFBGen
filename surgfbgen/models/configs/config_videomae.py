"""
Project: Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
"""
import torch
import torchvision.transforms as transforms
from transformers import VideoMAEConfig

config = dict(
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize frames to 224x224 as expected by VideoMAE
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ]),
    
    model_config = VideoMAEConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        num_frames=16,
        tubelet_size=2,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        qkv_bias=True,
        use_mean_pooling=True,
        decoder_num_attention_heads=6,
        decoder_hidden_size=384,
        decoder_num_hidden_layers=4,
        decoder_intermediate_size=1536,
        norm_pix_loss=True,
    ),
)
