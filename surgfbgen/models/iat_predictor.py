import os
import json
import time as time_module
import datetime
from typing import List, Union, Dict
from dataclasses import dataclass, field
from collections import OrderedDict
from enum import Enum
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
from pydantic import BaseModel

# Assuming these modules exist in your project structure
from surgfbgen.models.utils import load_surgical_VL_model
from surgfbgen.logger.logger import get_logger
from surgfbgen.prompts.chatllm_interface import ChatLLMInterface
from surgfbgen.prompts.base import prompt_library, PromptTemplate
from surgfbgen.logger.logger import EnhancedLogger

# --- Configuration Models ---

@dataclass
class IATPredictorType:
    """Defines the types of predictors available."""
    SURGICAL_VL: str = "surgical_vl"
    GPT: str = "gpt"

class PredictionType(str, Enum):
    """Defines the architecture and training strategy for Surgical VL models."""
    INDIVIDUAL = "individual"
    SIMULTANEOUS = "simultaneous"

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

class IATPredictorConfig(BaseModel):
    """Base configuration for any IAT predictor."""
    type: str
    aux_text_names: List[str]
    IAT_class_names: Dict[str, List[str]] = {
        'instrument': [], 'action': [], 'tissue': []
    }

class IATPredictorSurgicalVLConfig(IATPredictorConfig):
    """Configuration specific to the Surgical VL-based predictor."""
    type: str = IATPredictorType.SURGICAL_VL
    prediction_type: PredictionType = PredictionType.SIMULTANEOUS # New field
    surgical_VL_model_name: str = 'PeskaVLP'
    aux_text_names: List[str] = ['procedure', 'task']
    ffn_config: FFNConfig = FFNConfig()
    IAT_clf_head_configs: Dict[str, ClassificationHeadConfig] = {
        'instrument': ClassificationHeadConfig(num_classes=10, dropout=0.1),
        'action': ClassificationHeadConfig(num_classes=10, dropout=0.1),
        'tissue': ClassificationHeadConfig(num_classes=10, dropout=0.1)
    }

class IATPredictorGPTConfig(IATPredictorConfig):
    """Configuration specific to the GPT-based predictor."""
    type: str = IATPredictorType.GPT
    chatllm_config: ChatLLMConfig = ChatLLMConfig()
    aux_text_names: List[str] = ['procedure', 'task']
    prompt_name: str = "surgical_iat_predictor"

class IATPredictorTrainConfig(BaseModel):
    """Base configuration for training any IAT predictor."""
    pass

class IATPredictorSurgicalVLTrainConfig(IATPredictorTrainConfig):
    """Configuration specific to training the Surgical VL model."""
    freeze_surgical_vl: bool = True # training_type is removed
    optimizer_class: str = "AdamW"
    initial_learning_rate: float = 1e-4
    learning_rate_type: str = "cosine"
    warmup_ratio: float = 0.1
    max_iters: int = 10000
    n_iter_no_change: int = 500
    loss_fn: str = "CrossEntropyLoss"
    output_dir: str
    val_interval: int = 100

class IATPredictorGPTTrainConfig(IATPredictorTrainConfig):
    """Configuration for training/fine-tuning the GPT model (placeholder)."""
    pass


# --- Main Predictor Class ---

class SurgicalVLModel(nn.Module):
    """
    Encapsulates the Surgical VL pipeline.

    - If aux_text_names == [] (or no usable text is provided), it runs VIDEO-ONLY:
      pooled video embedding -> FFN/heads.
    - Otherwise, it runs VIDEO+TEXT:
      pooled video embedding concat with per-aux-text embeddings -> FFN/heads.

    Outputs always include:
      - {instrument_logits, action_logits, tissue_logits}
      - img_emb:  [B, T, Dv]
      - text_embs: Dict[name] -> [B, Dt] (empty dict in video-only mode)
      - concat_embs: [B, Dv] in video-only, or [B, Dv + sum(Dt)] in video+text
    """
    def __init__(self, surgical_vl_base, ffn, clf_heads, prediction_type):
        super().__init__()
        self.surgical_vl_base = surgical_vl_base
        self.ffn = ffn  # None if INDIVIDUAL
        self.clf_heads = clf_heads
        self.prediction_type = prediction_type

    @staticmethod
    def _to_pil(frame_tensor: torch.Tensor) -> Image.Image:
        """
        Accepts [H,W,C] or [C,H,W] as torch tensor (uint8 or float in [0,1]).
        Returns a PIL image.
        """
        x = frame_tensor
        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected frame as torch.Tensor")
        x = x.detach().cpu()
        if x.ndim != 3:
            raise ValueError(f"Frame must be 3D, got shape {tuple(x.shape)}")
        # CHW -> HWC
        if x.shape[0] in (1, 3):  # assume CHW
            x = x.permute(1, 2, 0)
        # float -> uint8
        if x.dtype.is_floating_point:
            x = (x.clamp(0, 1) * 255).to(torch.uint8)
        elif x.dtype != torch.uint8:
            x = x.to(torch.uint8)
        return Image.fromarray(x.numpy())

    def forward(self, video_frames, text_inputs, img_preprocess, tokenizer, aux_text_names):
        """
        video_frames: torch.Tensor [B, T, H, W, C] or [B, T, C, H, W] uint8/float
        text_inputs:  List[Dict[str,str]] or None
        """
        # Device resolution
        if isinstance(video_frames, torch.Tensor):
            device = video_frames.device
        else:
            device = next(self.parameters()).device

        B = video_frames.shape[0]

        # -----------------------------
        # 1) Video embeddings
        # -----------------------------
        with torch.no_grad():
            img_embs_list = []
            for i in range(B):
                frames = video_frames[i]  # [T, ...]
                # Convert each frame to PIL and apply model's img_preprocess
                processed = torch.stack([
                    img_preprocess(self._to_pil(frames[t]))
                    for t in range(frames.shape[0])
                ]).to(device)  # [T, 3, H', W']
                out = self.surgical_vl_base(processed, None, mode="video")
                img_emb = out["img_emb"]  # [T, Dv]
                img_embs_list.append(img_emb)

            img_embs = torch.stack(img_embs_list, dim=0)  # [B, T, Dv]
            pooled_img = img_embs.mean(dim=1)             # [B, Dv]

        # -----------------------------
        # 2) (Optional) Text embeddings
        # -----------------------------
        use_text = bool(aux_text_names)
        text_embs_stacked = {}
        combined_embs = pooled_img

        if use_text:
            # infer text emb dim for zero-padding if any sample lacks a field
            text_dim = self.surgical_vl_base.backbone_text.model.pooler.dense.out_features
            # init containers
            for name in aux_text_names:
                text_embs_stacked[name] = []

            for i in range(B):
                per_sample = text_inputs[i] if (text_inputs is not None and i < len(text_inputs) and isinstance(text_inputs[i], dict)) else {}
                for name in aux_text_names:
                    txt = per_sample.get(name, None)
                    if txt is None or txt == "":
                        text_embs_stacked[name].append(torch.zeros(text_dim, device=device))
                    else:
                        tokens = tokenizer([txt], device=device)
                        with torch.no_grad():
                            t_out = self.surgical_vl_base(None, tokens, mode="text")
                        # t_out['text_emb']: [1, Dt]
                        text_embs_stacked[name].append(t_out["text_emb"].squeeze(0))

            # stack per name -> [B, Dt]
            for name in aux_text_names:
                text_embs_stacked[name] = torch.stack(text_embs_stacked[name], dim=0)

            # concat in the given order
            full_text = torch.cat([text_embs_stacked[name] for name in aux_text_names], dim=-1)  # [B, sum(Dt)]
            combined_embs = torch.cat([pooled_img, full_text], dim=-1)  # [B, Dv + sum(Dt)]

        # -----------------------------
        # 3) Heads
        # -----------------------------
        if self.prediction_type == PredictionType.SIMULTANEOUS:
            ffn_out = self.ffn(combined_embs)
            outputs = {f"{key}_logits": head(ffn_out) for key, head in self.clf_heads.items()}
            outputs["ffn_out"] = ffn_out
        else:
            outputs = {f"{key}_logits": head(combined_embs) for key, head in self.clf_heads.items()}

        outputs.update({
            "img_emb": img_embs,
            "text_embs": text_embs_stacked,   # {} in video-only
            "concat_embs": combined_embs,     # pooled_img in video-only
        })
        return outputs



class IATPredictor(nn.Module):
    def __init__(
        self,
        iat_predictor_config: Union[IATPredictorSurgicalVLConfig, IATPredictorGPTConfig],
        device: str = "cuda",
        verbose: bool = True,
        logger: EnhancedLogger = None
    ):
        super().__init__()
        self.config = iat_predictor_config
        self.device = device
        self.verbose = verbose
        
        log_dir = os.path.join(os.environ.get('REPO_DIRECTORY', '.'), 'surgfbgen', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'iat_predictor-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.log')
        if logger:
            self.logger = logger
        else:
            self.logger = get_logger(name="IATPredictorLogger", log_file=log_file, console_output=self.verbose)
        
        self.model = None
        self.img_preprocess = None
        self.tokenizer = None
        self.chat_llm_interface = None
        self.system_prompt_template = None
        
        self._log("Validating configuration...")
        self._validate_config()
        self._log("Configuration validated.")
        
        self._log(f"Initializing predictor with type: '{self.config.type}'")
        if self.config.type == IATPredictorType.SURGICAL_VL:
            self._init_surgical_vl_predictor()
        elif self.config.type == IATPredictorType.GPT:
            self._init_gpt_predictor()
        else:
            raise ValueError(f"Unsupported IAT predictor type: {self.config.type}")

    def _log(self, message: str):
        self.logger.info(message)
    
    def _validate_config(self):
        if self.config.type == IATPredictorType.SURGICAL_VL:
            iat_names = self.config.IAT_class_names
            clf_configs = self.config.IAT_clf_head_configs
            for key in iat_names:
                if key not in clf_configs:
                    raise ValueError(f"Missing classification head config for: {key}")
                if len(iat_names[key]) != clf_configs[key].num_classes:
                    raise ValueError(f"Mismatch in class count for '{key}'")
            if self.config.surgical_VL_model_name not in SupportedModels().surgical_VL:
                raise ValueError(f"Unsupported surgical VL model: {self.config.surgical_VL_model_name}")
        
        elif self.config.type == IATPredictorType.GPT:
            if self.config.chatllm_config.model_name not in SupportedModels().chat_llm:
                raise ValueError(f"Unsupported ChatLLM model: {self.config.chatllm_config.model_name}")
            if not prompt_library.get(self.config.prompt_name):
                raise ValueError(f"Prompt '{self.config.prompt_name}' not found in prompt library.")

    def _init_surgical_vl_predictor(self):
        self._log(f"Loading base surgical VL model: {self.config.surgical_VL_model_name}")
        surgical_vl_base, self.img_preprocess, self.tokenizer = load_surgical_VL_model(
            model_name=self.config.surgical_VL_model_name, device=self.device)
        self._log("Base model loaded.")

        v_emb_dim = surgical_vl_base.backbone_img.global_embedder.out_features
        l_emb_dim = surgical_vl_base.backbone_text.model.pooler.dense.out_features
        total_emb_dim = v_emb_dim + len(self.config.aux_text_names) * l_emb_dim
        self._log(f"Total embedding dimension: {total_emb_dim}")

        ffn_config = self.config.ffn_config
        
        # Architectural split for initialization
        if self.config.prediction_type == PredictionType.SIMULTANEOUS:
            self._log("Building SIMULTANEOUS model with shared FFN.")
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
        else: # INDIVIDUAL
            self._log("Building INDIVIDUAL model with separate FFNs per head.")
            ffn = None # No shared FFN
            clf_heads = nn.ModuleDict()
            for key, cfg in self.config.IAT_clf_head_configs.items():
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
                head_layers[f'clf_linear'] = nn.Linear(in_features, cfg.num_classes)
                clf_heads[key] = nn.Sequential(head_layers)

        self.model = SurgicalVLModel(surgical_vl_base, ffn, clf_heads, self.config.prediction_type).to(self.device)
        self._log("Assembled unified SurgicalVLModel.")
    
    def _init_gpt_predictor(self):
        self._log("Initializing GPT-based predictor.")
        cfg = self.config.chatllm_config
        self.chat_llm_interface = ChatLLMInterface(
            model_name=cfg.model_name,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens
        )
        self.system_prompt_template = prompt_library.get(self.config.prompt_name)
        self._log(f"ChatLLMInterface initialized with model: '{cfg.model_name}' and prompt: '{self.config.prompt_name}'")
    
    def _forward_surgical_vl(self, video_frames: torch.Tensor, text_inputs: List[Dict[str, str]] = None):
        """Performs a forward pass using the unified Surgical VL model."""
        return self.model(
            video_frames,
            text_inputs,
            self.img_preprocess,
            self.tokenizer,
            self.config.aux_text_names
        )

    def _forward_gpt(self, video_frames: torch.Tensor, text_inputs: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
        batch_size = video_frames.shape[0]
        all_predictions = []

        iat_values_str = ""
        for key, values in self.config.IAT_class_names.items():
            iat_values_str += f"- Allowed \"{key}\" values: {json.dumps(values)}\n"
        system_prompt = self.system_prompt_template.format(iat_values=iat_values_str.strip())

        for i in range(batch_size):
            self._log(f"Processing batch item {i+1}/{batch_size}")
            
            frames = video_frames[i]
            np_frames = [(f.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8') for f in frames]
            
            aux_text = ""
            if text_inputs and i < len(text_inputs):
                for name, text in text_inputs[i].items():
                    if name in self.config.aux_text_names:
                        aux_text += f"{name.capitalize()}: {text}\n"
            
            user_prompt = [f"Analyze the following information and video frames:\n{aux_text}"] + np_frames

            parsed_response = None
            for attempt in range(self.config.chatllm_config.num_attempts):
                self._log(f"Attempt {attempt+1}/{self.config.chatllm_config.num_attempts} for batch item {i+1}")
                try:
                    raw_response = self.chat_llm_interface.generate(system_prompt, user_prompt)
                    if raw_response.strip().startswith("```json"):
                        raw_response = raw_response.strip()[7:-3].strip()
                    
                    parsed_response = json.loads(raw_response)
                    
                    if all(key in parsed_response for key in self.config.IAT_class_names.keys()):
                        self._log(f"Successfully parsed response for batch item {i+1}")
                        break
                    else:
                        self._log(f"Warning: Parsed JSON missing required keys. Response: {parsed_response}")
                        parsed_response = None
                except (json.JSONDecodeError, Exception) as e:
                    self._log(f"Warning: Failed to parse or validate LLM response on attempt {attempt+1}. Error: {e}")
            
            if parsed_response:
                all_predictions.append(parsed_response)
            else:
                self._log(f"Error: Failed to get a valid response for batch item {i+1} after all attempts.")
                all_predictions.append({key: "ERROR" for key in self.config.IAT_class_names.keys()})

        return all_predictions

    def forward(self, video_frames: torch.Tensor, text_inputs: List[Dict[str, str]] = None):
        if self.config.type == IATPredictorType.SURGICAL_VL:
            return self._forward_surgical_vl(video_frames, text_inputs)
        elif self.config.type == IATPredictorType.GPT:
            return self._forward_gpt(video_frames, text_inputs)
        else:
            raise ValueError(f"Unsupported IAT predictor type: {self.config.type}")

    def train(
        self,
        train_config: Union[IATPredictorSurgicalVLTrainConfig, IATPredictorGPTTrainConfig],
        train_dataloader: DataLoader,
        val_dataloader: DataLoader
    ):
        if self.config.type == IATPredictorType.SURGICAL_VL:
            self._train_surgical_vl(train_config, train_dataloader, val_dataloader)
        elif self.config.type == IATPredictorType.GPT:
            self._train_gpt(train_config, train_dataloader, val_dataloader)
        else:
            raise ValueError(f"Training not implemented for type: {self.config.type}")

    def _train_gpt(self, train_config: IATPredictorGPTTrainConfig, train_dataloader: DataLoader, val_dataloader: DataLoader):
        self._log("GPT fine-tuning is not yet implemented.")
        pass

    def _train_surgical_vl(self, train_config: IATPredictorSurgicalVLTrainConfig, train_dataloader: DataLoader, val_dataloader: DataLoader):
        loss_function = getattr(nn, train_config.loss_fn)()
        
        if train_config.freeze_surgical_vl:
            self._log("Training: surgical_vl_base is frozen.")
            for param in self.model.surgical_vl_base.parameters():
                param.requires_grad = False
        
        if self.config.prediction_type == PredictionType.SIMULTANEOUS:
            self._log("Training: FFN and all classification heads simultaneously.")
            optimizer = getattr(torch.optim, train_config.optimizer_class)(
                filter(lambda p: p.requires_grad, self.model.parameters()), lr=train_config.initial_learning_rate
            )
            self._run_train_loop(train_config, train_dataloader, val_dataloader, loss_function, optimizer)
        
        elif self.config.prediction_type == PredictionType.INDIVIDUAL:
            self._log("Training: classification heads individually.")
            for head_name in self.model.clf_heads.keys():
                self._log(f"--- Starting individual training for head: {head_name} ---")
                # Optimizer for only one sequential head (which includes its own FFN)
                optimizer = getattr(torch.optim, train_config.optimizer_class)(
                    filter(lambda p: p.requires_grad, self.model.clf_heads[head_name].parameters()),
                    lr=train_config.initial_learning_rate
                )
                self._run_train_loop(train_config, train_dataloader, val_dataloader, loss_function, optimizer, training_head=head_name)

    def _run_train_loop(self, train_config, train_dataloader, val_dataloader, loss_function, optimizer, training_head=None):
        best_val_loss = float('inf')
        iters_since_best = 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config.max_iters)
        data_iter = iter(train_dataloader)

        pbar = tqdm(
            range(train_config.max_iters),
            desc=f"Train ({training_head or 'simultaneous'})",
            dynamic_ncols=True,
            disable=not self.verbose,
            leave=True,
        )

        for i in pbar:
            try:
                batch = next(data_iter)
                if batch.get("_invalid_batch", False):
                    # nothing valid in this batch, skip safely
                    # if you want, log once per N occurrences
                    continue

                # optional: log if some samples were dropped in this batch
                num_dropped = batch.get("num_dropped", 0)
                if num_dropped > 0:
                    # prefer pbar.write if you're inside tqdm
                    # pbar.write(f"Dropped {num_dropped} samples this batch: {batch.get('dropped_cvids', [])}")
                    self._log(f"Dropped {num_dropped} samples this batch")
            except StopIteration:
                data_iter = iter(train_dataloader)
                batch = next(data_iter)

            video_frames = batch['video_frames'].to(self.device)
            text_inputs = batch['text_inputs']
            labels = {k: v.to(self.device) for k, v in batch['labels'].items()}

            self.model.train()
            outputs = self.forward(video_frames, text_inputs)

            optimizer.zero_grad()
            if training_head:
                loss = loss_function(outputs[f'{training_head}_logits'], labels[training_head])
            else:  # SIMULTANEOUS
                loss = sum(loss_function(outputs[f'{key}_logits'], labels[key]) for key in self.config.IAT_class_names.keys())

            loss.backward()
            optimizer.step()
            scheduler.step()

            # live progress bar info
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

            if i > 0 and i % 100 == 0:
                self._log(f"Iter {i}/{train_config.max_iters} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

            if i > 0 and i % train_config.val_interval == 0:
                val_loss = self._run_validation(val_dataloader, loss_function, training_head)
                if val_loss != val_loss:  # NaN check
                    self._log("Validation skipped (no valid batches).")
                    pass
                        
                self._log(f"--- Validation at Iter {i} | Val Loss: {val_loss:.4f} ---")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    iters_since_best = 0
                    os.makedirs(train_config.output_dir, exist_ok=True)
                    save_path = os.path.join(train_config.output_dir, f"best_model_{training_head or 'simultaneous'}.pt")
                    torch.save(self.model.state_dict(), save_path)
                    self._log(f"New best model saved to {save_path}")
                else:
                    iters_since_best += 1

            if iters_since_best * train_config.val_interval >= train_config.n_iter_no_change:
                self._log(f"Early stopping at iter {i} due to no improvement in validation loss.")
                break

        pbar.close()

    
    def _run_validation(self, val_dataloader, loss_function, training_head=None):
        self.model.eval()
        total_val_loss = 0.0
        valid_batches = 0

        with torch.no_grad():
            for batch in val_dataloader:
                if batch.get("_invalid_batch", False):
                    # skip batches with no readable samples
                    continue

                video_frames = batch['video_frames'].to(self.device)
                text_inputs = batch['text_inputs']
                labels = {k: v.to(self.device) for k, v in batch['labels'].items()}

                outputs = self.forward(video_frames, text_inputs)

                if training_head:
                    loss = loss_function(outputs[f'{training_head}_logits'], labels[training_head])
                else:
                    loss = sum(loss_function(outputs[f'{key}_logits'], labels[key]) for key in self.config.IAT_class_names.keys())

                total_val_loss += loss.item()
                valid_batches += 1

        if valid_batches == 0:
            # no valid batches this round; signal to caller to skip metrics update
            return float('nan')

        return total_val_loss / valid_batches
