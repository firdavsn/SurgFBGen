import os
import json
import time as time_module
import datetime
from typing import List, Union, Dict, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
from enum import Enum
from tqdm.auto import tqdm
import pandas as pd

import numpy as np
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pydantic import BaseModel

# Assuming these modules exist in your project structure
# Mock implementations are provided at the end for standalone execution
from surgfbgen.models.utils import load_surgical_VL_model
from surgfbgen.logger.logger import get_logger
from surgfbgen.prompts.chatllm_interface import ChatLLMInterface
from surgfbgen.prompts.base import prompt_library, PromptTemplate
from surgfbgen.logger.logger import EnhancedLogger

# --- Configuration Models ---

@dataclass
class IATPredictorType:
    """Defines the types of predictors available."""
    SURGICAL_VL_SIMULTANEOUS: str = "surgical_vl_simultaneous"
    SURGICAL_VL_INDIVIDUAL: str = "surgical_vl_individual"
    GPT: str = "gpt"

@dataclass
class SupportedModels:
    """Holds lists of supported model names for validation."""
    chat_llm: List[str] = field(default_factory=lambda: [
        'gpt-5', 'gpt-5-mini', 'gpt-4o', 'gpt-4o-mini',
        'gpt-4.1', 'gpt-4.1-mini', 'gemini-2.5-pro', 'gemini-2.5-flash'
    ])
    surgical_VL: List[str] = field(default_factory=lambda: ['SurgVLP', 'HecVL', 'PeskaVLP'])

class FFNConfig(BaseModel):
    """Configuration for the Feed-Forward Network."""
    hidden_sizes: List[int] = [128, 64]
    activation: str = "ReLU"
    dropout: float = 0.1

class ClassificationHeadConfig(BaseModel):
    """Configuration for a single classification head."""
    num_classes: int
    dropout: float = 0.1

class ChatLLMConfig(BaseModel):
    """Configuration for the ChatLLM interface."""
    model_name: str = 'gpt-4o-mini'
    temperature: float = 0.2
    max_tokens: int = 1000
    num_attempts: int = 3

# --- Base and GPT Configs ---

class IATPredictorConfigBase(BaseModel):
    """Base configuration for any IAT predictor."""
    type: str
    aux_text_names: List[str] = ['procedure', 'task']

class IATPredictorGPTConfig(IATPredictorConfigBase):
    """Configuration specific to the GPT-based predictor."""
    type: str = IATPredictorType.GPT
    chatllm_config: ChatLLMConfig = ChatLLMConfig()
    iat_definitions_path: str
    aux_text_names: List[str] = ['procedure', 'task']
    prompt_name: str = "surgical_iat_predictor"

# --- SIMULTANEOUS Predictor Configs ---

class IATSimultaneousPredictorConfig(IATPredictorConfigBase):
    """Configuration specific to the Simultaneous Surgical VL-based predictor."""
    type: str = IATPredictorType.SURGICAL_VL_SIMULTANEOUS
    surgical_VL_model_name: str = 'PeskaVLP'
    ffn_config: FFNConfig = FFNConfig()
    IAT_class_names: Dict[str, List[str]]
    IAT_clf_head_configs: Dict[str, ClassificationHeadConfig]

class IATSimultaneousPredictorTrainConfig(BaseModel):
    """Configuration specific to training the Simultaneous Surgical VL model."""
    freeze_surgical_vl: bool = True
    optimizer_class: str = "AdamW"
    initial_learning_rate: float = 1e-4
    learning_rate_type: str = "cosine"
    warmup_ratio: float = 0.1
    max_iters: int = 10000
    n_iter_no_change: int = 500
    loss_fn: str = "CrossEntropyLoss"
    output_dir: str
    val_interval: int = 100

# --- INDIVIDUAL Predictor Configs ---

class IATIndividualPredictorConfig(IATPredictorConfigBase):
    """Configuration specific to the Individual Surgical VL-based predictor."""
    type: str = IATPredictorType.SURGICAL_VL_INDIVIDUAL
    iat_target: str # e.g., 'instrument', 'action', or 'tissue'
    surgical_VL_model_name: str = 'PeskaVLP'
    ffn_config: FFNConfig = FFNConfig()
    class_names: List[str]
    clf_head_config: ClassificationHeadConfig

class IATIndividualPredictorTrainConfig(BaseModel):
    """Configuration specific to training the Individual Surgical VL model."""
    freeze_surgical_vl: bool = True
    optimizer_class: str = "AdamW"
    initial_learning_rate: float = 1e-4
    learning_rate_type: str = "cosine"
    warmup_ratio: float = 0.1
    max_iters: int = 10000
    n_iter_no_change: int = 500
    loss_fn: str = "CrossEntropyLoss"
    output_dir: str
    val_interval: int = 100


# --- Base Utility Functions ---

def _to_pil(frame_tensor: torch.Tensor) -> Image.Image:
    """Converts a torch tensor to a PIL image."""
    x = frame_tensor
    if not isinstance(x, torch.Tensor):
        raise TypeError("Expected frame as torch.Tensor")
    x = x.detach().cpu()
    if x.ndim != 3:
        raise ValueError(f"Frame must be 3D, got shape {tuple(x.shape)}")
    if x.shape[0] in (1, 3):  # CHW -> HWC
        x = x.permute(1, 2, 0)
    if x.dtype.is_floating_point:
        x = (x.clamp(0, 1) * 255).to(torch.uint8)
    elif x.dtype != torch.uint8:
        x = x.to(torch.uint8)
    return Image.fromarray(x.numpy())

def _get_combined_embeddings(
    surgical_vl_base: nn.Module,
    video_frames: torch.Tensor,
    text_inputs: List[Dict[str, str]],
    img_preprocess,
    tokenizer,
    aux_text_names: List[str],
    device: str
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Helper to extract and combine video and text embeddings."""
    B = video_frames.shape[0]

    # Video embeddings
    with torch.no_grad():
        img_embs_list = []
        for i in range(B):
            frames = video_frames[i]
            processed = torch.stack([img_preprocess(_to_pil(frames[t])) for t in range(frames.shape[0])]).to(device)
            out = surgical_vl_base(processed, None, mode="video")
            img_embs_list.append(out["img_emb"])
        img_embs = torch.stack(img_embs_list, dim=0)  # [B, T, Dv]
        pooled_img = img_embs.mean(dim=1)            # [B, Dv]

    # Text embeddings
    text_embs_stacked = {}
    combined_embs = pooled_img

    if aux_text_names:
        text_dim = surgical_vl_base.backbone_text.model.pooler.dense.out_features
        for name in aux_text_names:
            text_embs_stacked[name] = []

        for i in range(B):
            per_sample = text_inputs[i] if text_inputs and i < len(text_inputs) else {}
            for name in aux_text_names:
                txt = per_sample.get(name)
                if not txt:
                    text_embs_stacked[name].append(torch.zeros(text_dim, device=device))
                else:
                    tokens = tokenizer([txt], device=device)
                    with torch.no_grad():
                        t_out = surgical_vl_base(None, tokens, mode="text")
                    text_embs_stacked[name].append(t_out["text_emb"].squeeze(0))

        for name in aux_text_names:
            text_embs_stacked[name] = torch.stack(text_embs_stacked[name], dim=0)

        full_text = torch.cat([text_embs_stacked[name] for name in aux_text_names], dim=-1)
        combined_embs = torch.cat([pooled_img, full_text], dim=-1)

    return combined_embs, img_embs, text_embs_stacked


# --- SIMULTANEOUS Prediction Architecture ---

class SurgicalVLModelSimultaneous(nn.Module):
    """Encapsulates the SIMULTANEOUS Surgical VL pipeline: combined embeddings -> shared FFN -> multiple heads."""
    def __init__(self, surgical_vl_base, ffn, clf_heads):
        super().__init__()
        self.surgical_vl_base = surgical_vl_base
        self.ffn = ffn
        self.clf_heads = clf_heads

    def forward(self, video_frames, text_inputs, img_preprocess, tokenizer, aux_text_names):
        device = next(self.parameters()).device
        combined_embs, img_embs, text_embs_stacked = _get_combined_embeddings(
            self.surgical_vl_base, video_frames, text_inputs, img_preprocess, tokenizer, aux_text_names, device
        )

        ffn_out = self.ffn(combined_embs)
        outputs = {f"{key}_logits": head(ffn_out) for key, head in self.clf_heads.items()}
        
        outputs.update({
            "img_emb": img_embs,
            "text_embs": text_embs_stacked,
            "concat_embs": combined_embs,
            "ffn_out": ffn_out
        })
        return outputs

# --- INDIVIDUAL Prediction Architecture ---

class SurgicalVLModelIndividual(nn.Module):
    """Encapsulates the INDIVIDUAL Surgical VL pipeline: combined embeddings -> dedicated FFN+Head."""
    def __init__(self, surgical_vl_base, head):
        super().__init__()
        self.surgical_vl_base = surgical_vl_base
        self.head = head

    def forward(self, video_frames, text_inputs, img_preprocess, tokenizer, aux_text_names):
        device = next(self.parameters()).device
        combined_embs, img_embs, text_embs_stacked = _get_combined_embeddings(
            self.surgical_vl_base, video_frames, text_inputs, img_preprocess, tokenizer, aux_text_names, device
        )

        logits = self.head(combined_embs)
        
        return {
            "logits": logits,
            "img_emb": img_embs,
            "text_embs": text_embs_stacked,
            "concat_embs": combined_embs,
        }

# --- Main Predictor Classes ---

class IATSimultaneousPredictor(nn.Module):
    """
    Predicts Instrument, Action, and Tissue simultaneously using a shared 
    vision-language backbone and a common Feed-Forward Network (FFN) before
    splitting into separate classification heads.
    """
    def __init__(
        self, 
        config: 'IATSimultaneousPredictorConfig', 
        device: str = "cuda", 
        verbose: bool = True,
        logger: 'EnhancedLogger' = None  # <-- CORRECTED: Added optional logger argument
    ):
        """
        Initializes the simultaneous predictor.

        Args:
            config (IATSimultaneousPredictorConfig): Configuration object for the predictor.
            device (str): The device to run the model on ('cuda' or 'cpu').
            verbose (bool): If True, enables detailed logging to the console.
            logger (EnhancedLogger, optional): An external logger to use. If None, a new one is created.
        """
        super().__init__()
        self.config = config
        self.device = device
        self.verbose = verbose
        
        # --- CORRECTED: Logic to handle the optional logger ---
        if logger:
            self.logger = logger
        else:
            log_dir = os.path.join(os.environ.get('REPO_DIRECTORY', '.'), 'surgfbgen', 'logs')
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f'iat_simultaneous_predictor-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.log')
            self.logger = get_logger("IATSimultaneousPredictor", log_file=log_file, console_output=self.verbose)
        # ---------------------------------------------------------
        
        self.logger.info("Validating configuration...")
        self._validate_config()
        self.logger.info("Configuration validated.")

        self.logger.info("Initializing predictor model...")
        self._init_predictor()
        self.logger.info("Predictor initialized successfully.")
    
    # ... The rest of the class methods (_validate_config, _init_predictor, train, etc.) remain the same ...
    def _validate_config(self):
        """Ensures the provided configuration is valid."""
        # from .mock_definitions import SupportedModels
        if self.config.surgical_VL_model_name not in SupportedModels().surgical_VL:
            raise ValueError(f"Unsupported surgical VL model: {self.config.surgical_VL_model_name}")
        
        iat_names = self.config.IAT_class_names
        clf_configs = self.config.IAT_clf_head_configs
        
        if iat_names.keys() != clf_configs.keys():
            raise ValueError("IAT_class_names and IAT_clf_head_configs must have the same keys.")

        for key in iat_names:
            if len(iat_names[key]) != clf_configs[key].num_classes:
                raise ValueError(
                    f"Mismatch in class count for '{key}': "
                    f"{len(iat_names[key])} names vs {clf_configs[key].num_classes} classes in config."
                )

    def _init_predictor(self):
        """Loads the base model and constructs the prediction heads."""
        self.logger.info(f"Loading base surgical VL model: {self.config.surgical_VL_model_name}")
        # from .mock_definitions import load_surgical_VL_model
        surgical_vl_base, self.img_preprocess, self.tokenizer = load_surgical_VL_model(
            model_name=self.config.surgical_VL_model_name, device=self.device
        )
        self.logger.info("Base model loaded.")

        v_emb_dim = surgical_vl_base.backbone_img.global_embedder.out_features
        l_emb_dim = surgical_vl_base.backbone_text.model.pooler.dense.out_features
        total_emb_dim = v_emb_dim + len(self.config.aux_text_names) * l_emb_dim
        self.logger.info(f"Calculated embedding dimensions: Video({v_emb_dim}), Text({l_emb_dim} x {len(self.config.aux_text_names)}), Total({total_emb_dim})")

        self.logger.info("Building SIMULTANEOUS model with a shared FFN.")
        ffn_config = self.config.ffn_config
        in_features = total_emb_dim
        ffn_layers = OrderedDict()
        
        for i, hidden_size in enumerate(ffn_config.hidden_sizes):
            ffn_layers[f'linear{i}'] = nn.Linear(in_features, hidden_size)
            ffn_layers[f'{ffn_config.activation}{i}'] = nn.__dict__[ffn_config.activation]()
            in_features = hidden_size
        
        if ffn_config.dropout > 0:
            ffn_layers['dropout'] = nn.Dropout(ffn_config.dropout)
        
        ffn = nn.Sequential(ffn_layers)
        
        clf_heads = nn.ModuleDict({
            key: nn.Linear(in_features, cfg.num_classes)
            for key, cfg in self.config.IAT_clf_head_configs.items()
        })
        
        self.model = SurgicalVLModelSimultaneous(surgical_vl_base, ffn, clf_heads).to(self.device)
        self.logger.info("Assembled unified SurgicalVLModelSimultaneous.")

    def forward(self, video_frames: torch.Tensor, text_inputs: List[Dict[str, str]] = None):
        return self.model(
            video_frames,
            text_inputs,
            self.img_preprocess,
            self.tokenizer,
            self.config.aux_text_names
        )

    def train(
        self,
        train_config: 'IATSimultaneousPredictorTrainConfig',
        train_dataloader: DataLoader,
        val_dataloader: DataLoader
    ):
        config_msg = [
            '', '=' * 60, " Starting Training Run ", " Training Config:"
        ] + [
            f"   {key}: {str(value)}" for key, value in train_config.model_dump().items()
        ] + ['=' * 60]
        self.logger.info("\n".join(config_msg))
        
        self._train_surgical_vl(train_config, train_dataloader, val_dataloader)

    def _train_surgical_vl(
        self, 
        train_config: 'IATSimultaneousPredictorTrainConfig', 
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader
    ):
        loss_function = getattr(nn, train_config.loss_fn)()
        
        if train_config.freeze_surgical_vl:
            self.logger.info("Freezing weights of the base surgical_vl model.")
            for param in self.model.surgical_vl_base.parameters():
                param.requires_grad = False
        
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = getattr(torch.optim, train_config.optimizer_class)(
            trainable_params, lr=train_config.initial_learning_rate
        )
        
        self.logger.info("Starting training loop for the FFN and all classification heads.")
        self._run_train_loop(train_config, train_dataloader, val_dataloader, loss_function, optimizer)

    def _run_train_loop(
        self, 
        train_config: 'IATSimultaneousPredictorTrainConfig', 
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader, 
        loss_function, 
        optimizer
    ):
        best_val_loss = float('inf')
        iters_since_best = 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config.max_iters)
        data_iter = iter(train_dataloader)

        pbar = tqdm(
            range(train_config.max_iters),
            desc="Train (Simultaneous)",
            dynamic_ncols=True,
            disable=not self.verbose,
        )

        for i in pbar:
            try:
                batch = next(data_iter)
                if batch.get("_invalid_batch", False):
                    self.logger.info("Skipping an entirely invalid batch from dataloader.")
                    continue
            except StopIteration as e:
                self.logger.info(f"DataLoader exhausted at iteration {i}. Error: {e}")
                self.logger.info("Resetting training data iterator.")
                data_iter = iter(train_dataloader)
                batch = next(data_iter)

            video_frames = batch['video_frames'].to(self.device)
            text_inputs = batch['text_inputs']
            labels = {k: v.to(self.device) for k, v in batch['labels'].items()}

            self.model.train()
            outputs = self.forward(video_frames, text_inputs)

            optimizer.zero_grad()
            
            loss = sum(
                loss_function(outputs[f'{key}_logits'], labels[key]) 
                for key in self.config.IAT_class_names.keys()
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

            if i > 0 and i % train_config.val_interval == 0:
                val_loss = self._run_validation(val_dataloader, loss_function)
                if val_loss != val_loss:
                    self.logger.warning("Validation returned NaN. Skipping metrics update for this interval.")
                    continue
                
                self.logger.info(f"--- Validation at Iter {i} | Val Loss: {val_loss:.4f} ---")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    iters_since_best = 0
                    os.makedirs(train_config.output_dir, exist_ok=True)
                    save_path = os.path.join(train_config.output_dir, "best_model_simultaneous.pt")
                    torch.save(self.model.state_dict(), save_path)
                    self.logger.info(f"New best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")
                else:
                    iters_since_best += 1

            if iters_since_best * train_config.val_interval >= train_config.n_iter_no_change:
                self.logger.info(f"Early stopping at iter {i} due to no improvement in validation loss for {train_config.n_iter_no_change} iterations.")
                break
        
        pbar.close()
        self.logger.info("Training loop finished.")

    def _run_validation(self, val_dataloader: DataLoader, loss_function) -> float:
        self.model.eval()
        total_val_loss = 0.0
        valid_batches = 0

        with torch.no_grad():
            for batch in val_dataloader:
                if batch.get("_invalid_batch", False):
                    continue

                video_frames = batch['video_frames'].to(self.device)
                text_inputs = batch['text_inputs']
                labels = {k: v.to(self.device) for k, v in batch['labels'].items()}

                outputs = self.forward(video_frames, text_inputs)
                
                loss = sum(
                    loss_function(outputs[f'{key}_logits'], labels[key]) 
                    for key in self.config.IAT_class_names.keys()
                )

                total_val_loss += loss.item()
                valid_batches += 1

        if valid_batches == 0:
            self.logger.warning("Validation run completed with zero valid batches.")
            return float('nan')

        return total_val_loss / valid_batches


class IATIndividualPredictor(nn.Module):
    """
    Predicts a single IAT category (e.g., Instrument, Action, or Tissue)
    using a shared vision-language backbone and a dedicated Feed-Forward
    Network (FFN) and classification head for that specific target.
    """
    def __init__(
        self,
        config: 'IATIndividualPredictorConfig',
        device: str = "cuda",
        verbose: bool = True,
        logger: 'EnhancedLogger' = None  # <-- Added optional logger argument
    ):
        """
        Initializes the individual predictor.

        Args:
            config (IATIndividualPredictorConfig): Configuration object for the predictor.
            device (str): The device to run the model on ('cuda' or 'cpu').
            verbose (bool): If True, enables detailed logging to the console.
            logger (EnhancedLogger, optional): An external logger to use. If None, a new one is created.
        """
        super().__init__()
        self.config = config
        self.device = device
        self.verbose = verbose
        
        # --- Logic to handle the optional logger ---
        if logger:
            self.logger = logger
        else:
            log_dir = os.path.join(os.environ.get('REPO_DIRECTORY', '.'), 'surgfbgen', 'logs')
            os.makedirs(log_dir, exist_ok=True)
            
            # Make logger name and file specific to the target
            target_name = self.config.iat_target.capitalize()
            logger_name = f"IATIndividualPredictor-{target_name}"
            log_file_name = f'iat_individual_predictor_{self.config.iat_target}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.log'
            log_file = os.path.join(log_dir, log_file_name)
            
            self.logger = get_logger(logger_name, log_file=log_file, console_output=self.verbose)
        # ---------------------------------------------------------
        
        self.logger.info("Validating configuration...")
        self._validate_config()
        self.logger.info("Configuration validated.")

        self.logger.info("Initializing predictor model...")
        self._init_predictor()
        self.logger.info("Predictor initialized successfully.")

    def _validate_config(self):
        """Ensures the provided configuration is valid."""
        # from .mock_definitions import SupportedModels
        if self.config.surgical_VL_model_name not in SupportedModels().surgical_VL:
            raise ValueError(f"Unsupported surgical VL model: {self.config.surgical_VL_model_name}")

        if len(self.config.class_names) != self.config.clf_head_config.num_classes:
            raise ValueError(
                f"Mismatch in class count for target '{self.config.iat_target}': "
                f"{len(self.config.class_names)} names vs {self.config.clf_head_config.num_classes} classes in config."
            )

    def _init_predictor(self):
        """Loads the base model and constructs the dedicated prediction head."""
        self.logger.info(f"Loading base surgical VL model: {self.config.surgical_VL_model_name}")
        # from .mock_definitions import load_surgical_VL_model
        surgical_vl_base, self.img_preprocess, self.tokenizer = load_surgical_VL_model(
            model_name=self.config.surgical_VL_model_name, device=self.device
        )
        self.logger.info("Base model loaded.")

        v_emb_dim = surgical_vl_base.backbone_img.global_embedder.out_features
        l_emb_dim = surgical_vl_base.backbone_text.model.pooler.dense.out_features
        total_emb_dim = v_emb_dim + len(self.config.aux_text_names) * l_emb_dim
        self.logger.info(f"Calculated embedding dimensions: Video({v_emb_dim}), Text({l_emb_dim} x {len(self.config.aux_text_names)}), Total({total_emb_dim})")

        self.logger.info(f"Building INDIVIDUAL model for target: '{self.config.iat_target}'")
        ffn_config = self.config.ffn_config
        in_features = total_emb_dim
        
        head_layers = OrderedDict()
        # Create an independent FFN for this head
        for i, hidden_size in enumerate(ffn_config.hidden_sizes):
            head_layers[f'ffn_linear{i}'] = nn.Linear(in_features, hidden_size)
            head_layers[f'ffn_{ffn_config.activation}{i}'] = nn.__dict__[ffn_config.activation]()
            in_features = hidden_size
        
        if ffn_config.dropout > 0:
            head_layers['ffn_dropout'] = nn.Dropout(ffn_config.dropout)
        
        # Add the final classification layer
        head_layers['clf_linear'] = nn.Linear(in_features, self.config.clf_head_config.num_classes)
        
        head = nn.Sequential(head_layers)
        
        self.model = SurgicalVLModelIndividual(surgical_vl_base, head).to(self.device)
        self.logger.info(f"Assembled unified SurgicalVLModelIndividual for '{self.config.iat_target}'.")

    def forward(self, video_frames: torch.Tensor, text_inputs: List[Dict[str, str]] = None):
        """
        Performs a forward pass through the model.

        Args:
            video_frames (torch.Tensor): A batch of video frames.
            text_inputs (List[Dict[str, str]], optional): A batch of auxiliary text inputs.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the logits for the target category
                                     and intermediate embeddings.
        """
        return self.model(
            video_frames,
            text_inputs,
            self.img_preprocess,
            self.tokenizer,
            self.config.aux_text_names
        )

    def train(
        self,
        train_config: 'IATIndividualPredictorTrainConfig',
        train_dataloader: DataLoader,
        val_dataloader: DataLoader
    ):
        """
        Main training entry point for the individual predictor.

        Args:
            train_config (IATIndividualPredictorTrainConfig): Configuration for the training run.
            train_dataloader (DataLoader): DataLoader for the training set.
            val_dataloader (DataLoader): DataLoader for the validation set.
        """
        config_msg = [
            '', '=' * 60, f" Starting Training Run for '{self.config.iat_target.upper()}' ", " Training Config:"
        ] + [
            f"   {key}: {str(value)}" for key, value in train_config.model_dump().items()
        ] + ['=' * 60]
        self.logger.info("\n".join(config_msg))
        
        self._train_surgical_vl(train_config, train_dataloader, val_dataloader)

    def _train_surgical_vl(
        self,
        train_config: 'IATIndividualPredictorTrainConfig',
        train_dataloader: DataLoader,
        val_dataloader: DataLoader
    ):
        """Internal training handler for the Surgical VL model."""
        loss_function = getattr(nn, train_config.loss_fn)()
        
        if train_config.freeze_surgical_vl:
            self.logger.info("Freezing weights of the base surgical_vl model.")
            for param in self.model.surgical_vl_base.parameters():
                param.requires_grad = False
        
        # The optimizer will only see the parameters of the head, which includes the FFN
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = getattr(torch.optim, train_config.optimizer_class)(
            trainable_params, lr=train_config.initial_learning_rate
        )
        
        self.logger.info(f"Starting training loop for the '{self.config.iat_target}' head.")
        self._run_train_loop(train_config, train_dataloader, val_dataloader, loss_function, optimizer)

    def _run_train_loop(
        self,
        train_config: 'IATIndividualPredictorTrainConfig',
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        loss_function,
        optimizer
    ):
        """The core training and validation loop."""
        best_val_loss = float('inf')
        iters_since_best = 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config.max_iters)
        data_iter = iter(train_dataloader)

        pbar = tqdm(
            range(train_config.max_iters),
            desc=f"Train ({self.config.iat_target.capitalize()})",
            dynamic_ncols=True,
            disable=not self.verbose,
        )

        for i in pbar:
            try:
                batch = next(data_iter)
                if batch.get("_invalid_batch", False):
                    self.logger.warning("Skipping an entirely invalid batch from dataloader.")
                    continue
            except StopIteration:
                self.logger.info("Resetting training data iterator.")
                data_iter = iter(train_dataloader)
                batch = next(data_iter)

            video_frames = batch['video_frames'].to(self.device)
            text_inputs = batch['text_inputs']
            # For the individual predictor, 'labels' is a single tensor
            labels = batch['labels'].to(self.device)

            self.model.train()
            outputs = self.forward(video_frames, text_inputs)

            optimizer.zero_grad()
            
            # Loss is calculated on the single 'logits' output
            loss = loss_function(outputs['logits'], labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

            if i > 0 and i % train_config.val_interval == 0:
                val_loss = self._run_validation(val_dataloader, loss_function)
                if val_loss != val_loss:  # NaN check
                    self.logger.warning("Validation returned NaN. Skipping metrics update for this interval.")
                    continue
                
                self.logger.info(f"--- Validation at Iter {i} | Val Loss: {val_loss:.4f} ---")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    iters_since_best = 0
                    os.makedirs(train_config.output_dir, exist_ok=True)
                    save_path = os.path.join(train_config.output_dir, f"best_model_{self.config.iat_target}.pt")
                    torch.save(self.model.state_dict(), save_path)
                    self.logger.info(f"New best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")
                else:
                    iters_since_best += 1

            if iters_since_best * train_config.val_interval >= train_config.n_iter_no_change:
                self.logger.info(f"Early stopping at iter {i} due to no improvement in validation loss for {train_config.n_iter_no_change} iterations.")
                break
        
        pbar.close()
        self.logger.info("Training loop finished.")

    def _run_validation(self, val_dataloader: DataLoader, loss_function) -> float:
        """Runs a single validation epoch."""
        self.model.eval()
        total_val_loss = 0.0
        valid_batches = 0

        with torch.no_grad():
            for batch in val_dataloader:
                if batch.get("_invalid_batch", False):
                    continue

                video_frames = batch['video_frames'].to(self.device)
                text_inputs = batch['text_inputs']
                labels = batch['labels'].to(self.device)

                outputs = self.forward(video_frames, text_inputs)
                
                loss = loss_function(outputs['logits'], labels)
                
                total_val_loss += loss.item()
                valid_batches += 1

        if valid_batches == 0:
            self.logger.warning("Validation run completed with zero valid batches.")
            return float('nan')

        return total_val_loss / valid_batches

class IATGPTPredictor:
    """
    Predicts Instrument, Action, and Tissue (IAT) by leveraging a 
    multimodal Large Language Model (e.g., GPT-4o).

    This class is not a torch.nn.Module as it does not involve gradient-based
    training within this script. It acts as an interface to an external API.
    """
    def __init__(
        self,
        config: 'IATPredictorGPTConfig',
        verbose: bool = True
    ):
        """
        Initializes the GPT-based predictor.

        Args:
            config (IATPredictorGPTConfig): Configuration for the GPT predictor.
            verbose (bool): If True, enables detailed logging to the console.
        """
        self.config = config
        self.verbose = verbose
        
        log_dir = os.path.join(os.environ.get('REPO_DIRECTORY', '.'), 'surgfbgen', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file_name = f'iat_gpt_predictor-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.log'
        log_file = os.path.join(log_dir, log_file_name)
        
        self.logger = get_logger("IATGPTPredictor", log_file=log_file, console_output=self.verbose)
        
        self.logger.info("Validating configuration...")
        self._validate_config()
        self.logger.info("Configuration validated.")

        self.logger.info("Initializing predictor interface...")
        self._init_predictor()
        self.logger.info("Predictor interface initialized successfully.")

    def _validate_config(self):
        """Ensures the provided configuration is valid."""
        # from .mock_definitions import SupportedModels, prompt_library
        if self.config.chatllm_config.model_name not in SupportedModels().chat_llm:
            raise ValueError(f"Unsupported ChatLLM model: {self.config.chatllm_config.model_name}")
        
        if not prompt_library.get(self.config.prompt_name):
            raise ValueError(f"Prompt '{self.config.prompt_name}' not found in prompt library.")

    def _init_predictor(self):
        """Initializes the ChatLLM interface and loads the prompt template."""
        self.logger.info("Initializing GPT-based predictor.")
        cfg = self.config.chatllm_config
        
        # from .mock_definitions import ChatLLMInterface, prompt_library
        self.chat_llm_interface = ChatLLMInterface(
            model_name=cfg.model_name,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens
        )
        self.system_prompt_template = prompt_library.get(self.config.prompt_name)
        
        self.logger.info(f"ChatLLMInterface initialized with model: '{cfg.model_name}' and prompt: '{self.config.prompt_name}'")

    def predict(self, video_frames: torch.Tensor, text_inputs: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """
        Generates IAT predictions for a batch of video and text inputs.

        Args:
            video_frames (torch.Tensor): A tensor of video frames with shape 
                                         [B, T, C, H, W] or [B, T, H, W, C].
            text_inputs (List[Dict[str, str]], optional): A list of dictionaries, 
                                                          where each dictionary contains 
                                                          auxiliary text data for a sample.

        Returns:
            List[Dict[str, str]]: A list of dictionaries, each containing the 
                                  predicted 'instrument', 'action', and 'tissue'.
        """
        batch_size = video_frames.shape[0]
        all_predictions = []

        # Prepare the static part of the system prompt
        iat_values_str = ""
        for key, values in self.config.IAT_class_names.items():
            iat_values_str += f"- Allowed \"{key}\" values: {json.dumps(values)}\n"
        system_prompt = self.system_prompt_template.format(iat_values=iat_values_str.strip())

        for i in range(batch_size):
            self.logger.info(f"Processing batch item {i+1}/{batch_size}")
            
            # Convert tensor frames to a list of PIL Images for the LLM interface
            frames_for_sample = video_frames[i] # [T, C, H, W] or [T, H, W, C]
            pil_frames = []
            for frame_tensor in frames_for_sample:
                # Ensure tensor is on CPU and in HWC format for conversion
                x = frame_tensor.detach().cpu()
                if x.shape[0] in (1, 3): # CHW -> HWC
                    x = x.permute(1, 2, 0)
                if x.dtype.is_floating_point:
                    x = (x.clamp(0, 1) * 255)
                x = x.to(torch.uint8).numpy()
                pil_frames.append(Image.fromarray(x))

            # Prepare the user prompt content (text + images)
            aux_text = ""
            if text_inputs and i < len(text_inputs):
                for name, text in text_inputs[i].items():
                    if name in self.config.aux_text_names and text:
                        aux_text += f"{name.capitalize()}: {text}\n"
            
            user_prompt_content = [f"Analyze the following information and video frames:\n{aux_text}"] + pil_frames

            parsed_response = None
            for attempt in range(self.config.chatllm_config.num_attempts):
                self.logger.info(f"Attempt {attempt+1}/{self.config.chatllm_config.num_attempts} for batch item {i+1}")
                try:
                    raw_response = self.chat_llm_interface.generate(system_prompt, user_prompt_content)
                    
                    # Clean the response to extract JSON
                    if "```json" in raw_response:
                        raw_response = raw_response.split("```json")[1].split("```")[0].strip()
                    
                    parsed_response = json.loads(raw_response)
                    
                    # Validate that all required keys are in the parsed response
                    if all(key in parsed_response for key in self.config.IAT_class_names.keys()):
                        self.logger.info(f"Successfully parsed valid response for batch item {i+1}")
                        break # Exit retry loop on success
                    else:
                        self.logger.warning(f"Parsed JSON missing required keys. Response: {parsed_response}")
                        parsed_response = None # Invalidate to trigger retry

                except (json.JSONDecodeError, Exception) as e:
                    self.logger.error(f"Failed to parse or validate LLM response on attempt {attempt+1}. Error: {e}\nRaw Response: '{raw_response}'")
            
            if parsed_response:
                all_predictions.append(parsed_response)
            else:
                self.logger.error(f"Failed to get a valid response for batch item {i+1} after all attempts.")
                all_predictions.append({key: "ERROR" for key in self.config.IAT_class_names.keys()})

        return all_predictions

    def train(self, *args, **kwargs):
        """
        Placeholder for training/fine-tuning.
        
        Fine-tuning large vision-language models is a complex, resource-intensive
        process that is typically handled by separate, specialized scripts and is
        out of scope for this predictor class.
        """
        self.logger.warning("Training/fine-tuning for the IATGPTPredictor is not implemented in this class.")
        self.logger.warning("This process should be handled by a dedicated fine-tuning pipeline.")
        pass

# --- Datasets and Collate Functions ---

def _load_and_sample_video(clip_path: str, num_frames: int, resize_hw: Tuple[int, int]) -> torch.Tensor:
    """Loads, samples, and resizes video frames."""
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video: {clip_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise ValueError("Video is empty")

    # Uniformly sample indices
    if total_frames <= num_frames:
        indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
    else:
        step = total_frames / float(num_frames)
        indices = [min(int(i * step), total_frames - 1) for i in range(num_frames)]
    
    frames = []
    for idx in sorted(list(set(indices))): # Read each unique frame only once
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            H, W = resize_hw
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            frames.append(frame)
    cap.release()

    # Create the final list by selecting from the read frames
    frame_map = {idx: frame for idx, frame in zip(sorted(list(set(indices))), frames)}
    sampled = [frame_map[i] for i in indices]

    return torch.from_numpy(np.stack(sampled, axis=0)).to(torch.uint8) # [T,H,W,C]

class IATSimultaneousPredictorDataset(Dataset):
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
        aux_text_names: List[str] = ["procedure", "task"]
    ):
        import pandas as pd  # local import to avoid global dependency
        self.df = pd.read_csv(csv_path)
        self.clips_dir = clips_dir
        self.num_frames = num_frames
        self.resize_hw = resize_hw
        self.clip_ext = clip_ext
        self.aux_text_names = aux_text_names
        
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
            frames = IATSimultaneousPredictorDataset.load_all_frames(clip_path)
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


        idxs = IATSimultaneousPredictorDataset.uniform_indices(len(frames), self.num_frames)
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
        text_inputs = {}
        for name in self.aux_text_names:
            text = row.get(name, "")
            if pd.isna(text):
                text = 'NONE'
            text_inputs[name] = text
        return {
            "original_dialogue": str(row["dialogue"]),
            "cvid": str(row["cvid"]),
            "video_frames": vid,  # [T,H,W,C], uint8
            "text_inputs": text_inputs,
            "labels": labels,
            "_invalid": False,
        }

def make_one_hot_collate_simultaneous(IAT_class_names: Dict[str, List[str]]):
    """Collate function for the simultaneous dataset."""
    idx_maps = {k: {cls: i for i, cls in enumerate(v)} for k, v in IAT_class_names.items()}
    sizes = {k: len(v) for k, v in IAT_class_names.items()}

    def _collate(samples):
        # filter out invalid samples from dataset
        valid = [s for s in samples if isinstance(s, dict) and not s.get("_invalid", False)]
        dropped = [s.get("cvid", None) for s in samples if isinstance(s, dict) and s.get("_invalid", False)]

        if len(valid) == 0:
            # return a small sentinel batch the train/val loops can detect
            print(f"Returning empty batch with {len(dropped)} dropped samples: {dropped}")
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

class IATIndividualPredictorDataset(Dataset):
    """
    Dataset for training an Individual IAT Predictor.
    Loads video clips and corresponding labels for a single IAT target.
    Includes logic to downsample the 'NONE' class.
    """
    def __init__(
        self,
        csv_path: str,
        clips_dir: str,
        iat_target: str,
        num_frames: int,
        resize_hw: Tuple[int, int],
        aux_text_names: List[str],
        downsample_none_ratio: float = 1.0,
        clip_ext: str = "avi",
        is_train_set: bool = False,
        seed: int = None
    ):
        self.logger = get_logger("IATIndividualDataset")
        self.original_df = pd.read_csv(csv_path)
        self.df = self.original_df.copy()
        
        self.clips_dir = clips_dir
        self.iat_target = iat_target
        self.num_frames = num_frames
        self.resize_hw = resize_hw
        self.clip_ext = clip_ext
        self.aux_text_names = aux_text_names
        self.seed = seed

        # Validate columns
        required_cols = ["cvid", iat_target] + aux_text_names
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column '{col}' in {csv_path}")

        # --- Downsampling Logic for 'NONE' class ---
        # Only apply to the training set
        if is_train_set and 'NONE' in self.df[self.iat_target].unique() and downsample_none_ratio is not None:
            self.logger.info(f"Applying 'NONE' class downsampling for target '{self.iat_target}' with ratio {downsample_none_ratio}.")
            
            df_none = self.df[self.df[self.iat_target] == 'NONE']
            df_others = self.df[self.df[self.iat_target] != 'NONE']

            if not df_others.empty:
                class_counts = df_others[self.iat_target].value_counts()
                # Get count of the most frequent class *besides* 'NONE'
                most_frequent_other_count = class_counts.iloc[0]
                
                target_none_count = int(most_frequent_other_count * downsample_none_ratio)
                original_none_count = len(df_none)

                self.logger.info(f"  - Original 'NONE' count: {original_none_count}")
                self.logger.info(f"  - Most frequent other class ('{class_counts.index[0]}') has count: {most_frequent_other_count}")
                self.logger.info(f"  - Target 'NONE' count set to: {target_none_count}")

                if original_none_count > target_none_count:
                    df_none_sampled = df_none.sample(n=target_none_count, random_state=self.seed)
                    self.df = pd.concat([df_others, df_none_sampled]).sample(frac=1, random_state=self.seed).reset_index(drop=True)
                    self.logger.info(f"  - Downsampled 'NONE' to {len(df_none_sampled)} samples. Total training samples now: {len(self.df)}")
                else:
                    self.logger.info("  - 'NONE' count is already at or below the target. No downsampling performed.")
            else:
                self.logger.warning("  - No non-'NONE' samples found. Skipping downsampling.")

    def get_classes(self) -> List[str]:
        """Gets all unique class names from the original (pre-downsampling) dataframe."""
        return sorted(self.original_df[self.iat_target].unique().tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        clip_name = str(row["cvid"])
        
        if not clip_name.lower().endswith(f".{self.clip_ext}"):
            clip_name = f"{clip_name}.{self.clip_ext}"
        clip_path = os.path.join(self.clips_dir, clip_name)

        try:
            video_frames = _load_and_sample_video(clip_path, self.num_frames, self.resize_hw)
        except (RuntimeError, ValueError) as e:
            self.logger.warning(f"Could not load video {clip_path}: {e}. Skipping sample.")
            return {"_invalid": True, "cvid": clip_name, "error": str(e)}

        label = str(row[self.iat_target])
        text_inputs = {name: str(row.get(name, "")) for name in self.aux_text_names}

        return {
            "video_frames": video_frames,
            "text_inputs": text_inputs,
            "label": label,
            "_invalid": False
        }

def make_one_hot_collate_individual(class_names: List[str]):
    """Collate function for the individual dataset."""
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    num_classes = len(class_names)
    
    def _collate(samples):
        valid = [s for s in samples if not s.get("_invalid")]
        dropped = [s.get("cvid", None) for s in samples if isinstance(s, dict) and s.get("_invalid", False)]

        if not valid: 
            print(f"Returning empty batch with {len(dropped)} dropped samples: {dropped}")
            return {"_invalid_batch": True, "num_dropped": len(dropped), "dropped_cvids": dropped}

        vids = torch.stack([s["video_frames"] for s in valid])
        texts = [s["text_inputs"] for s in valid]
        
        labels = torch.zeros(len(valid), len(class_names), dtype=torch.float32)
        for i, s in enumerate(valid):
            cls_name = s["label"]
            labels[i, class_to_idx[cls_name]] = 1.0
        
        # indices = torch.tensor([class_to_idx[s["label"]] for s in valid], dtype=torch.long)
        # labels = nn.functional.one_hot(indices, num_classes=num_classes).float()

        # return {"video_frames": vids, "text_inputs": texts, "labels": labels}
    
        return {
            "video_frames": vids,
            "text_inputs": texts,
            "labels": labels,
            "_invalid_batch": False,
            "num_dropped": len(dropped),
            "dropped_cvids": dropped[:8],  # cap the list so logs don't explode
        }
        
    return _collate