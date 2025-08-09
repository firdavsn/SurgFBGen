from pydantic import BaseModel
from typing import List, Union, Dict
from dataclasses import dataclass
import time as time_module
import datetime
import os
from torch import nn
from collections import OrderedDict
import torch
from PIL import Image

import surgvlp

from surgfbgen.models.utils import (
    load_surgical_VL_model,
    load_VideoMAE_model
)
from surgfbgen.logger.logger import EnhancedLogger, get_logger

@dataclass
class IATPredictorType:
    EMBEDDING: str = "embedding"
    GPT: str = "gpt"

@dataclass
class SupportedModels:
    chat_llm: List[str] = ['gpt-5-nano', 'gpt-5-mini', 'gpt-5', 'gemini-2.5-flash-lite', 'gemini-2.5-flash', 'gemini-2.5-pro']
    # video_encoder: List[str] = ['VideoMAE', 'SurgVLP', 'HecVL', 'PeskaVLP']
    # language_encoder: List[str] = ['MedEmbed', 'SurgVLP', 'HecVL', 'PeskaVLP']
    surgical_VL: List[str] = ['SurgVLP', 'HecVL', 'PeskaVLP']

class FFNConfig(BaseModel):
    hidden_sizes: List[int] = [128, 64]
    activation: str = "ReLU"
    dropout: float = 0.1
    
class ClassificationHeadConfig(BaseModel):
    num_classes: int
    dropout: float = 0.1

class ChatLLMConfig(BaseModel):
    model_name: str = 'gpt-5-mini'
    temperature: float = 0.2
    max_tokens: int = 1000
    num_attempts: int = 3

class IATPredictorConfig(BaseModel):
    type: IATPredictorType
    aux_text_names: List[str]
    IAT_class_names: Dict[str, List[str]] = {
        'instrument': [], 'action': [], 'tissue': []
    }

class IATPredictorConfigEmbedding(IATPredictorConfig):
    type: IATPredictorType = IATPredictorType.EMBEDDING
    surgical_VL_model_name: str = 'PeskaVLP'
    aux_text_names: List[str] = ['procedure', 'task']
    fnn_config: FFNConfig = FFNConfig()
    IAT_clf_head_configs: Dict[str, ClassificationHeadConfig] = {
        'instrument': ClassificationHeadConfig(num_classes=10, dropout=0.1),
        'action': ClassificationHeadConfig(num_classes=10, dropout=0.1),
        'tissue': ClassificationHeadConfig(num_classes=10, dropout=0.1)
    }

class IATPredictorConfigGPT(IATPredictorConfig):
    type: IATPredictorType = IATPredictorType.GPT
    chatllm_config: ChatLLMConfig = ChatLLMConfig()
    aux_text_names: List[str] = ['procedure', 'task']

class IATPredictor(nn.Module):
    def __init__(
        self,
        iat_predictor_config: Union[IATPredictorConfig, IATPredictorConfigEmbedding, IATPredictorConfigGPT],
        device: str = "cuda",
        verbose: bool = True
    ):
        self.config = iat_predictor_config
        self.device = device
        self.verbose = verbose
        self._start_time = time_module.time()
        self.logger = get_logger(
            name="IATPredictorLogger",
            log_file=os.path.join(
                os.environ['REPO_DIRECTORY'], 
                'surgfbgen', 
                'logs', 
                f'iat_predictor-{datetime.datetime.now().strftime('%Y%M%D-%H:%M:%S')}.log'
            ),
            flush_every=5,
            console_output=self.verbose
        )
        
        self.models = None
        self.img_preprocess = None
        self.tokenizer = None
        
        self._log("Validating IAT Predictor configuration...")
        self._validate_config()
        self._log("Configuration validated successfully.")
        
        self._log(f"Initializing IAT Predictor with config: {self.config}")
        if self.config.type == IATPredictorType.EMBEDDING:
            self._init_embedding_predictor()
        elif self.config.type == IATPredictorType.GPT:
            self._init_gpt_predictor()
        else:
            raise ValueError(f"Unsupported IAT predictor type: {self.config.type}")

    def _log(self, message: str):
        self.logger.info(message)
    
    def _validate_config(self):
        if self.config.type == IATPredictorType.EMBEDDING:
            # Ensure that the IAT class names and classification head configs are consistent
            IAT_class_names = self.config.IAT_class_names
            clf_head_configs = self.config.IAT_clf_head_configs
            for key in IAT_class_names:
                if key not in clf_head_configs:
                    raise ValueError(f"Missing classification head config for: {key}")
                if len(IAT_class_names[key]) != clf_head_configs[key].num_classes:
                    raise ValueError(f"Mismatch in number of classes for {key}: "
                                    f"{len(IAT_class_names[key])} vs {clf_head_configs[key].num_classes}")
                    
            # Ensure model name is supported
            if self.config.surgical_VL_model_name not in SupportedModels.surgical_VL:
                raise ValueError(f"Unsupported surgical VL model: {self.config.surgical_VL_model_name}")
    
    def _init_embedding_predictor(self):
        model_name = self.config.surgical_VL_model_name
        
        self._log(f"Loading surgical VL model: {model_name}")
        surgical_VL_model, self.img_preprocess, self.tokenizer = load_surgical_VL_model(
            model_name=model_name,
            device=self.device
        )
        self._log(f"Surgical VL model loaded successfully: {model_name}")
        
        v_emb_dim = surgical_VL_model.backbone_img.global_embedder.out_features             # 768 for PeskaVLP
        l_emb_dim = surgical_VL_model.backbone_text.model.pooler.dense.out_features   # 768 for PeskaVLP
        total_emb_dim = v_emb_dim + len(self.config.aux_text_names) * l_emb_dim
        self._log(f"Total embedding dimension: {total_emb_dim} (v_emb_dim: {v_emb_dim}, l_emb_dim: {l_emb_dim})")
        
        fnn_config = self.config.fnn_config
        hidden_sizes = fnn_config.hidden_sizes
        activation_fn = fnn_config.activation
        dropout = fnn_config.dropout
        
        fnn_layers = OrderedDict()
        in_features = total_emb_dim
        for i, hidden_size in enumerate(fnn_config.hidden_sizes):
            fnn_layers['linear' + str(i)] = nn.Linear(in_features, hidden_size)
            fnn_layers[activation_fn + str(i)] = nn.__dict__[activation_fn]()
            in_features = hidden_size
        if dropout > 0:
            fnn_layers['dropout'] = nn.Dropout(dropout)
        fnn = nn.Sequential(fnn_layers)
        fnn_out_features = hidden_sizes[-1]
        self._log(f"Initialized FNN.")
        
        clf_head_configs = self.config.IAT_clf_head_configs
        clf_heads = nn.ModuleDict({
            'instrument': nn.Linear(in_features=fnn_out_features, out_features=clf_head_configs['instrument'].num_classes),
            'action': nn.Linear(in_features=fnn_out_features, out_features=clf_head_configs['action'].num_classes),
            'tissue': nn.Linear(in_features=fnn_out_features, out_features=clf_head_configs['tissue'].num_classes)
        })
        self._log(f"Initialized classification heads: {list(clf_heads.keys())}")
        
        self.models = nn.ModuleDict({
            'surgical_VL_model': surgical_VL_model,
            'fnn': fnn,
            'clf_heads': clf_heads
        })
        
    def _init_gpt_predictor(self):
        pass
    
    def _forward_embedding(
        self, 
        video_frames: torch.Tensor,                 # shape: (b, total_frames, 3, H, W)
        text_inputs: List[Dict[str, str]] = None    # shape: (b, ); keys must match aux_text_names
    ):
        batch_size, total_frames, _, H, W = video_frames.shape
        device = video_frames.device
        
        surgical_VL_model = self.models['surgical_VL_model']
        fnn = self.models['fnn']
        clf_heads = self.models['clf_heads']
        
        # Vision embedding
        img_embs = []
        for i in range(batch_size):
            frames = video_frames[i]  # shape: (total_frames, 3, H, W)
            processed_frames = []
            for frame in frames:
                processed_frame = self.img_preprocess(Image.fromarray(frame.cpu().numpy()))
                processed_frames.append(processed_frame)
            processed_frames = torch.stack(processed_frames).to(self.device)
            with torch.no_grad():
                img_emb = surgical_VL_model(processed_frames, None, mode='video')['img_emb']  # (total_frames, img_emb_dim)
            img_embs.append(img_emb.cpu().detach().numpy())
        img_embs = torch.stack(img_embs).to(device)
                
        # Text embedding
        text_embs = {name: [] for name in self.config.aux_text_names}
        for i in range(batch_size):
            inputs = text_inputs[i]
            for name, text in inputs.items():
                if name in self.config.aux_text_names:
                    with torch.no_grad():
                        tokens = self.tokenizer([text], device=device)
                        text_emb = surgical_VL_model(None, text , mode='text')['text_emb']
                        text_embs[name].append(text_emb.cpu().detach().numpy())
                else:
                    raise ValueError(f"Unsupported text input name: {name}")  
        for name in text_embs:
            text_embs[name] = torch.stack(text_embs[name]).to(device)  # shape: (b, text_emb_dim)
            
        # Concatenate embeddings
        embs = []
        for i in range(batch_size):
            img_emb = img_embs[i]  # shape: (total_frames, img_emb_dim)
            text_emb = torch.cat([text_embs[name][i] for name in self.config.aux_text_names], dim=-1)  # shape: (text_emb_dim * num_aux_texts)
            combined_emb = torch.cat([img_emb.flatten(start_dim=0, end_dim=1), text_emb], dim=-1)  # shape: (total_frames * img_emb_dim + text_emb_dim * num_aux_texts)
            embs.append(combined_emb)
        embs = torch.stack(embs).to(device)  # shape: (b, total_frames * img_emb_dim + text_emb_dim * num_aux_texts)
        
        # Forward through FNN
        fnn_out = fnn(embs)  # shape: (b, fnn_out_features)
        
        # Forward through classification heads
        outputs = {}
        for key in clf_heads:
            outputs[f'{key}_logits'] = clf_heads[key](fnn_out)  # shape: (b, num_classes)
        
        # Add intermediate outputs
        outputs['img_emb'] = img_embs  # shape: (b, total_frames, img_emb_dim)
        outputs['text_embs'] = text_embs  # dict of shape (b, text_emb_dim)
        outputs['concat_embs'] = embs  # shape: (b, total_frames * img_emb_dim + text_emb_dim * num_aux_texts)
        outputs['fnn_out'] = fnn_out
        
        return outputs
    
    def _forward_gpt(
        self, 
        video_frames: torch.Tensor, 
        text_inputs: Dict[str, str] = None  # keys must match aux_text_names
    ):
        pass
    
    def forward(
        self, 
        video_frames: torch.Tensor, 
        text_inputs: Dict[str, str] = None  # keys must match aux_text_names
    ):
        if self.config.type == IATPredictorType.EMBEDDING:
            return self._forward_embedding(video_frames, text_inputs)
        elif self.config.type == IATPredictorType.GPT:
            return self._forward_gpt(video_frames, text_inputs)
        else:
            raise ValueError(f"Unsupported IAT predictor type: {self.config.type}")