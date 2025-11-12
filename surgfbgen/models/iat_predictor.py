from pydantic import BaseModel
import pandas as pd
import os
import json
from typing import Tuple, List, Dict, Union
import numpy as np
import cv2
from tqdm import tqdm

from surgfbgen.prompts.chatllm_interface import ChatLLMInterface
from surgfbgen.prompts.cli import format_prompt

PROMPT_NAME = 'surgical_iat_predictor'

IAT_CLASS_DEFINITIONS_PATH = os.path.join(os.environ['REPO_DIR'], 'data', 'iat', 'iat_class_definitions.json')
with open(IAT_CLASS_DEFINITIONS_PATH, 'r') as f:
    IAT_CLASS_DEFINITIONS = json.load(f)

IAT_CLASS_DEFINITIONS['instrument']['UNCERTAIN'] = "The instrument is uncertain or not clearly identifiable."
IAT_CLASS_DEFINITIONS['action']['UNCERTAIN'] = "The action is uncertain or not clearly identifiable."
IAT_CLASS_DEFINITIONS['tissue']['UNCERTAIN'] = "The tissue is uncertain or not clearly identifiable."

class IATPredictorGPTConfig(BaseModel):
    input_frames: bool = True
    input_procedure: bool = True
    input_task: bool = True
    
    clips_dir: str
    
    chatllm_name: str
    temperature: float = 0.2
    max_tokens: int = 10_000
    
class IATGPTPredictor:
    def __init__(
        self,
        config: IATPredictorGPTConfig,
        api_key: str,   # either OpenAI or Gemini model. will be inferred from the model name
    ):
        self.config = config
        self.chatllm_interface = ChatLLMInterface(
            model_name=config.chatllm_name,
            api_key=api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    
    def _get_and_format_frames(
        self,
        cvid: str
    ):
        clip_path = os.path.join(self.config.clips_dir, cvid)  
        frames = _load_and_sample_video(
            clip_path=clip_path,
            resize_hw=(224, 224),  # Assuming a fixed size for simplicity
            num_frames=10
        )
        frames_list = []
        for frame in frames:
            frames_list.append(frame)
        return frames
    
    def _get_and_format_class_definitions(self) -> str:
        return json.dumps(IAT_CLASS_DEFINITIONS, indent=4)
    
    def _generate_iat_triplet(
        self,
        row: pd.Series,
        data_df: pd.DataFrame
    ) -> str:
        prompt_args = {}
        if self.config.input_procedure:
            procedure = row['procedure']
            procedure_defn = row['procedure_defn']
        else:
            procedure = 'N/A'
            procedure_defn = 'N/A'
            
        if self.config.input_task:
            task = row['task']
            task_defn = row['task_defn']
        else:
            task = 'N/A'
            task_defn = 'N/A'
        
        if self.config.input_frames:
            frames = self._get_and_format_frames(row['cvid'])
        else:
            frames = None
            
        
        class_definitions = self._get_and_format_class_definitions()
        
        
        prompt_args = {
            'procedure': procedure,
            'procedure_defn': procedure_defn,
            'task': task,
            'task_defn': task_defn,
            'iat_definitions': class_definitions,
        }
        prompt = format_prompt(
            name=PROMPT_NAME,
            kwargs=prompt_args
        )
        prompt = prompt.split('Video Frames:\n')[0]

        user_prompt = [prompt]
        if self.config.input_frames and frames is not None:
            user_prompt.append('Video Frames:\n')
            for frame in frames:
                user_prompt.append(frame)
        user_prompt.append('\n\nPredicted IAT Triplet (JSON Output):\n')
            
        response = self.chatllm_interface.generate(
            user_prompt=user_prompt
        )
        return response.strip()
    
    def generate_all_iat_triplets(
        self,
        data_df: pd.DataFrame, 
        override_existing: bool = False,
    ):
        # data_df.columns = [dialogue, cvid, procedure, procedure_defn, task, task_defn, (iat_triplet)]
        
        output_rows = []
        for i in tqdm(range(len(data_df)), desc="Generating IAT triplets"):
            row = data_df.iloc[i]
            row_dict = row.to_dict()
            if not override_existing and 'iat_triplet' in row_dict and not pd.isna(row_dict['iat_triplet']):
                print("Warning: 'iat_triplet' column already exists in the input DataFrame. Skipping.")
                output_rows.append(row_dict)
                continue
            iat_triplet = self._generate_iat_triplet(row, data_df)
            row_dict['iat_triplet'] = iat_triplet
            output_rows.append(row_dict)
        
        output_df = pd.DataFrame(output_rows)
        return output_df

def _load_and_sample_video(clip_path: str, resize_hw: Tuple[int, int], num_frames: int = None) -> np.ndarray:
    """Loads, samples, and resizes video frames."""
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video: {clip_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise ValueError("Video is empty")

    # Uniformly sample indices
    if num_frames is None:
        num_frames = total_frames
        
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

    return np.stack(sampled, axis=0) # [T,H,W,C]


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset


class HybridDataset(Dataset):
    """
    Dataset class for hybrid model training with embeddings, tracks, and labels.
    """
    def __init__(self, embeddings, tracks, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.tracks = torch.tensor(tracks, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.tracks[idx], self.labels[idx]


class FFNFusionModel(nn.Module):
    """
    A model that fuses a global embedding with a sequential track feature
    using a Feed-Forward Network (FFN) for classification.

    This model first processes the track and embedding features independently through
    two separate 2-layer MLPs. The track sequence is summarized by averaging its
    features before being passed to its processor. The outputs of these processors
    are then concatenated and passed through a final series of dense layers to
    produce the classification logits.
    """
    def __init__(
        self,
        embedding_dim,
        track_feature_dim,
        num_classes,
        track_summary_dim=64,
        embedding_summary_dim=64,
        hidden_dims=[128, 64],
        dropout=0.2
    ):
        """
        Initializes the FFNFusionModel.

        Args:
            embedding_dim (int): The dimensionality of the global input embedding.
            track_feature_dim (int): The dimensionality of features in the track sequence.
            num_classes (int): The number of output classes for classification.
            track_summary_dim (int): The output dimension for the track processing MLP.
            embedding_summary_dim (int): The output dimension for the embedding processing MLP.
            hidden_dims (list of int): A list specifying the size of each hidden layer 
                                       in the final classification head. Defaults to [128, 64].
            dropout (float): The dropout rate to apply after hidden layers. Defaults to 0.2.
        """
        super(FFNFusionModel, self).__init__()

        # MLP to process the track sequence and create a summary
        self.track_processor = nn.Sequential(
            nn.Linear(track_feature_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], track_summary_dim)
        )

        # MLP to process the global embedding
        self.embedding_processor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], embedding_summary_dim)
        )

        # Build the final classification head dynamically
        input_dim = track_summary_dim + embedding_summary_dim
        layers = []
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h_dim
        
        layers.append(nn.Linear(input_dim, num_classes))
        self.classification_head = nn.Sequential(*layers)

    def forward(self, embedding, track):
        """
        Forward pass of the model.

        Args:
            embedding (torch.Tensor): The global embedding tensor.
                                      Shape: (batch_size, embedding_dim)
            track (torch.Tensor): The sequence of track features.
                                  Shape: (batch_size, seq_len, track_feature_dim)

        Returns:
            torch.Tensor: The output logits from the classification head.
                          Shape: (batch_size, num_classes)
        """
        # Summarize track by averaging over sequence length
        track_mean = torch.mean(track, dim=1)  # (batch_size, track_feature_dim)
        track_summary = self.track_processor(track_mean)  # (batch_size, track_summary_dim)

        # Process the global embedding
        processed_embedding = self.embedding_processor(embedding)  # (batch_size, embedding_summary_dim)
        
        # Concatenate processed features
        combined_features = torch.cat((processed_embedding, track_summary), dim=1)
        
        # Get final predictions
        logits = self.classification_head(combined_features)
        return logits


class LSTMFusionModel(nn.Module):
    """
    A model that fuses a global embedding with sequential track features
    using an LSTM for classification.
    
    The track sequence is processed through an LSTM, and the final hidden state
    is concatenated with the global embedding before being passed through a
    classification head.
    """
    def __init__(
        self,
        embedding_dim,
        track_feature_dim,
        num_classes,
        lstm_hidden_dim=64,
        num_lstm_layers=1,
        dropout=0.2
    ):
        """
        Initializes the LSTMFusionModel.

        Args:
            embedding_dim (int): The dimensionality of the global input embedding.
            track_feature_dim (int): The dimensionality of features in the track sequence.
            num_classes (int): The number of output classes for classification.
            lstm_hidden_dim (int): The hidden dimension of the LSTM. Defaults to 64.
            num_lstm_layers (int): The number of LSTM layers. Defaults to 1.
            dropout (float): The dropout rate to apply in the classification head. Defaults to 0.2.
        """
        super(LSTMFusionModel, self).__init__()
        
        # LSTM to process the instrument track sequence
        self.lstm = nn.LSTM(
            input_size=track_feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0
        )
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.LayerNorm(embedding_dim + lstm_hidden_dim),
            nn.Linear(embedding_dim + lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes)
        )

    def forward(self, embedding, track):
        """
        Forward pass of the model.

        Args:
            embedding (torch.Tensor): The global embedding tensor.
                                      Shape: (batch_size, embedding_dim)
            track (torch.Tensor): The sequence of track features.
                                  Shape: (batch_size, seq_len, track_feature_dim)

        Returns:
            torch.Tensor: The output logits from the classification head.
                          Shape: (batch_size, num_classes)
        """
        # Process track with LSTM
        _, (hidden_state, _) = self.lstm(track)
        
        # Get the hidden state from the last layer
        track_summary = hidden_state[-1]  # (batch_size, lstm_hidden_dim)
        
        # Concatenate the global embedding with the track summary
        combined_features = torch.cat((embedding, track_summary), dim=1)
        
        # Get final predictions
        logits = self.classification_head(combined_features)
        return logits
    
    def extract_features(self, embedding, track):
        """
        Extract combined features without classification.
        
        Args:
            embedding (torch.Tensor): The global embedding tensor.
            track (torch.Tensor): The sequence of track features.
            
        Returns:
            torch.Tensor: Combined features before classification head.
        """
        _, (hidden_state, _) = self.lstm(track)
        track_summary = hidden_state[-1]
        combined_features = torch.cat((embedding, track_summary), dim=1)
        return combined_features


class CrossTransformerModel(nn.Module):
    """
    A model that uses cross-attention between embeddings and track features
    via a Transformer encoder for classification.
    
    The model projects both the embedding and track features into a common
    hidden dimension, adds a CLS token, applies positional encoding, and
    processes everything through a Transformer encoder.
    """
    def __init__(
        self,
        embedding_dim,
        track_feature_dim,
        num_heads,
        num_classes,
        hidden_dim=128,
        num_layers=1,
        dropout=0.2
    ):
        """
        Initializes the CrossTransformerModel.

        Args:
            embedding_dim (int): The dimensionality of the global input embedding.
            track_feature_dim (int): The dimensionality of features in the track sequence.
            num_heads (int): The number of attention heads in the Transformer.
            num_classes (int): The number of output classes for classification.
            hidden_dim (int): The hidden dimension for the Transformer. Defaults to 128.
            num_layers (int): The number of Transformer encoder layers. Defaults to 1.
            dropout (float): The dropout rate. Defaults to 0.2.
        """
        super(CrossTransformerModel, self).__init__()
        
        self.embedding_proj = nn.Linear(embedding_dim, hidden_dim)
        self.track_proj = nn.Linear(track_feature_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Sinusoidal positional embeddings
        max_len = 200
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(1, max_len, hidden_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_embedding', pe)
        
        self.pos_dropout = nn.Dropout(p=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, embedding, track):
        """
        Forward pass of the model.

        Args:
            embedding (torch.Tensor): The global embedding tensor.
                                      Shape: (batch_size, embedding_dim)
            track (torch.Tensor): The sequence of track features.
                                  Shape: (batch_size, seq_len, track_feature_dim)

        Returns:
            torch.Tensor: The output logits from the classification head.
                          Shape: (batch_size, num_classes)
        """
        batch_size = embedding.shape[0]
        
        # Project embeddings and tracks to hidden dimension
        emb_proj = self.embedding_proj(embedding).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        track_proj = self.track_proj(track)  # (batch_size, seq_len, hidden_dim)
        
        # Expand CLS token for the batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Concatenate CLS token (fused with embedding) and track features
        x = torch.cat((cls_tokens + emb_proj, track_proj), dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.pos_dropout(x)
        
        # Process through Transformer
        transformer_output = self.transformer_encoder(x)
        
        # Use CLS token output for classification
        cls_output = transformer_output[:, 0, :]
        
        logits = self.classification_head(cls_output)
        return logits


class PyTorchMLP(nn.Module):
    """
    A simple PyTorch MLP classifier that mirrors sklearn's MLPClassifier structure.
    This class is designed to be compatible with weights transferred from sklearn.
    """
    def __init__(self, input_dim, hidden_layers, num_classes):
        """
        Initializes the PyTorchMLP.

        Args:
            input_dim (int): The dimensionality of the input features.
            hidden_layers (list of int): A list of hidden layer sizes.
            num_classes (int): The number of output classes.
        """
        super(PyTorchMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input features. Shape: (batch_size, input_dim)

        Returns:
            torch.Tensor: Output logits. Shape: (batch_size, num_classes)
        """
        return self.network(x)


def sklearn_to_pytorch_mlp(sklearn_mlp, input_dim, num_classes, device='cpu'):
    """
    Converts a trained sklearn MLPClassifier to a PyTorch nn.Module with transferred weights.
    
    Args:
        sklearn_mlp (MLPClassifier): A trained sklearn MLPClassifier.
        input_dim (int): The input dimension of the MLP.
        num_classes (int): The number of output classes.
        device (str): Device to place the PyTorch model on. Defaults to 'cpu'.
    
    Returns:
        PyTorchMLP: A PyTorch MLP module with weights transferred from sklearn.
    """
    # Get hidden layer sizes from sklearn model
    hidden_layers = list(sklearn_mlp.hidden_layer_sizes)
    
    # Create PyTorch MLP
    pytorch_mlp = PyTorchMLP(input_dim, hidden_layers, num_classes).to(device)
    
    # Transfer weights
    # sklearn stores weights as coefs_ (list of weight matrices) and intercepts_ (list of bias vectors)
    with torch.no_grad():
        layer_idx = 0
        for i, (coef, intercept) in enumerate(zip(sklearn_mlp.coefs_, sklearn_mlp.intercepts_)):
            # sklearn uses (input_dim, output_dim) while PyTorch uses (output_dim, input_dim)
            # So we need to transpose
            weight = torch.tensor(coef.T, dtype=torch.float32).to(device)
            bias = torch.tensor(intercept, dtype=torch.float32).to(device)
            
            # Find the corresponding Linear layer in PyTorch model
            # Account for ReLU layers between Linear layers
            linear_layer = pytorch_mlp.network[layer_idx * 2]
            linear_layer.weight.copy_(weight)
            linear_layer.bias.copy_(bias)
            
            layer_idx += 1
    
    pytorch_mlp.eval()
    return pytorch_mlp


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import copy
import collections
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from torch.utils.data import DataLoader


class IATPredictor(nn.Module):
    """
    An individual model for IAT (Instrument, Action, or Tissue) prediction that consists of:
    1. A feature encoder (e.g., LSTMFusionModel) that processes embeddings and tracks
    2. An optional MLP classifier that operates on the extracted features
    
    This wrapper provides a unified interface for both end-to-end and decoupled training modes.
    """
    def __init__(
        self,
        feature_encoder,
        mlp_classifier=None,
        label_encoder=None,
        device='cpu'
    ):
        """
        Initializes the IATPredictor.

        Args:
            feature_encoder (nn.Module): The feature extraction model (e.g., LSTMFusionModel).
            mlp_classifier (nn.Module, optional): The MLP classifier. If None, uses end-to-end mode.
            label_encoder (LabelEncoder, optional): The label encoder for converting predictions to labels.
            device (str): Device to use. Defaults to 'cpu'.
        """
        super(IATPredictor, self).__init__()
        
        self.feature_encoder = feature_encoder.to(device)
        self.mlp_classifier = mlp_classifier.to(device) if mlp_classifier is not None else None
        self.label_encoder = label_encoder
        self.device = device
        self.decoupled = mlp_classifier is not None
        
    def forward(self, embedding, track):
        """
        Forward pass through the predictor.

        Args:
            embedding (torch.Tensor): The global embedding tensor.
                                      Shape: (batch_size, embedding_dim)
            track (torch.Tensor): The sequence of track features.
                                  Shape: (batch_size, seq_len, track_feature_dim)

        Returns:
            torch.Tensor: The output logits. Shape: (batch_size, num_classes)
        """
        if self.decoupled:
            # Extract features and pass through MLP
            with torch.no_grad():
                features = self.feature_encoder.extract_features(embedding, track)
            logits = self.mlp_classifier(features)
        else:
            # End-to-end prediction
            logits = self.feature_encoder(embedding, track)
        
        return logits
    
    def predict(self, embedding, track, return_probs=False):
        """
        Make predictions with the model.

        Args:
            embedding (torch.Tensor or np.ndarray): The global embedding tensor.
            track (torch.Tensor or np.ndarray): The sequence of track features.
            return_probs (bool): If True, return probabilities instead of class indices.

        Returns:
            np.ndarray: Predicted class indices or probabilities.
        """
        self.eval()
        
        # Convert to tensors if needed
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        if not isinstance(track, torch.Tensor):
            track = torch.tensor(track, dtype=torch.float32)
        
        # Ensure batch dimension
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        if track.dim() == 2:
            track = track.unsqueeze(0)
        
        embedding = embedding.to(self.device)
        track = track.to(self.device)
        
        with torch.no_grad():
            logits = self.forward(embedding, track)
            probs = torch.softmax(logits, dim=1)
        
        if return_probs:
            return probs.cpu().numpy()
        else:
            preds = torch.argmax(probs, dim=1)
            return preds.cpu().numpy()
    
    def predict_labels(self, embedding, track):
        """
        Make predictions and return the actual labels (if label_encoder is available).

        Args:
            embedding (torch.Tensor or np.ndarray): The global embedding tensor.
            track (torch.Tensor or np.ndarray): The sequence of track features.

        Returns:
            np.ndarray: Predicted labels.
        """
        preds = self.predict(embedding, track, return_probs=False)
        
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(preds)
        else:
            return preds
    
    def save(self, save_path):
        """
        Save the complete predictor model.

        Args:
            save_path (str): Path to save the model.
        """
        save_dict = {
            'feature_encoder_state_dict': self.feature_encoder.state_dict(),
            'decoupled': self.decoupled,
            'label_encoder': self.label_encoder,
        }
        
        if self.mlp_classifier is not None:
            save_dict['mlp_classifier_state_dict'] = self.mlp_classifier.state_dict()
            
        torch.save(save_dict, save_path)
        print(f"Saved IATPredictor to {save_path}")
    
    @classmethod
    def load(cls, load_path, feature_encoder_class, feature_encoder_kwargs, 
             mlp_input_dim=None, mlp_hidden_layers=None, num_classes=None, device='cpu'):
        """
        Load a saved IATPredictor model.

        Args:
            load_path (str): Path to the saved model.
            feature_encoder_class (class): The class of the feature encoder (e.g., LSTMFusionModel).
            feature_encoder_kwargs (dict): Keyword arguments to initialize the feature encoder.
            mlp_input_dim (int, optional): Input dimension for MLP (required if decoupled).
            mlp_hidden_layers (list, optional): Hidden layer sizes for MLP (required if decoupled).
            num_classes (int, optional): Number of classes (required if decoupled).
            device (str): Device to load the model on. Defaults to 'cpu'.

        Returns:
            IATPredictor: The loaded predictor model.
        """
        checkpoint = torch.load(load_path, map_location=device)
        
        # Initialize feature encoder
        feature_encoder = feature_encoder_class(**feature_encoder_kwargs)
        feature_encoder.load_state_dict(checkpoint['feature_encoder_state_dict'])
        
        # Initialize MLP if decoupled
        mlp_classifier = None
        if checkpoint['decoupled']:
            if mlp_input_dim is None or mlp_hidden_layers is None or num_classes is None:
                raise ValueError("mlp_input_dim, mlp_hidden_layers, and num_classes must be provided for decoupled models")
            
            mlp_classifier = PyTorchMLP(mlp_input_dim, mlp_hidden_layers, num_classes)
            mlp_classifier.load_state_dict(checkpoint['mlp_classifier_state_dict'])
        
        # Create predictor
        predictor = cls(
            feature_encoder=feature_encoder,
            mlp_classifier=mlp_classifier,
            label_encoder=checkpoint.get('label_encoder'),
            device=device
        )
        
        print(f"Loaded IATPredictor from {load_path}")
        return predictor
    
    @classmethod
    def train_model(
        cls,
        X_train_embs,
        X_train_tracks, 
        y_train,
        X_val_embs=None,
        X_val_tracks=None,
        y_val=None,
        feature_encoder_class=None,
        num_classes=None,
        label_encoder=None,
        decouple_mlp=False,
        epochs=20,
        batch_size=8,
        learning_rate=0.001,
        lstm_hidden_dim=32,
        num_lstm_layers=10,
        dropout=0.2,
        metric_avg='weighted',
        seed=42,
        device='cuda'
    ):
        """
        Train an IATPredictor model from prepared data.
        
        This method handles the actual training loop, model initialization,
        and optional MLP decoupling.
        
        Args:
            X_train_embs (np.ndarray): Training embeddings. Shape: (n_samples, embedding_dim)
            X_train_tracks (np.ndarray): Training tracks. Shape: (n_samples, seq_len, track_feature_dim)
            y_train (np.ndarray): Training labels (already encoded).
            X_val_embs (np.ndarray, optional): Validation embeddings.
            X_val_tracks (np.ndarray, optional): Validation tracks.
            y_val (np.ndarray, optional): Validation labels.
            feature_encoder_class (class): The feature encoder class to use (e.g., LSTMFusionModel).
            num_classes (int): Number of output classes.
            label_encoder (LabelEncoder): Label encoder for converting predictions.
            decouple_mlp (bool): Whether to train a separate MLP classifier.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size.
            learning_rate (float): Learning rate.
            lstm_hidden_dim (int): LSTM hidden dimension.
            num_lstm_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
            metric_avg (str): Averaging method for metrics.
            seed (int): Random seed.
            device (str): Device to use.
            
        Returns:
            tuple: (IATPredictor instance, metrics dict)
        """
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed_all(seed)
        
        # Create data loaders
        train_dataset = HybridDataset(X_train_embs, X_train_tracks, y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        
        use_validation = X_val_embs is not None and X_val_tracks is not None and y_val is not None
        val_loader = None
        if use_validation:
            val_dataset = HybridDataset(X_val_embs, X_val_tracks, y_val)
            val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        embedding_dim = X_train_embs.shape[1]
        track_feature_dim = X_train_tracks.shape[2]
        
        print(f"Embedding dim: {embedding_dim}, Track feature dim: {track_feature_dim}")
        print(f"Number of classes: {num_classes}")
        
        feature_encoder = feature_encoder_class(
            embedding_dim=embedding_dim,
            track_feature_dim=track_feature_dim,
            num_classes=num_classes,
            lstm_hidden_dim=lstm_hidden_dim,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(feature_encoder.parameters(), lr=learning_rate)
        
        best_val_loss = float('inf')
        best_model_state = None
        metrics = collections.defaultdict(list)
        
        # Training loop
        print(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            feature_encoder.train()
            train_loss = 0.0
            
            for batch_embs, batch_tracks, batch_labels in train_loader:
                batch_embs = batch_embs.to(device)
                batch_tracks = batch_tracks.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = feature_encoder(batch_embs, batch_tracks)
                loss = criterion(outputs, batch_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            metrics['train_loss'].append(avg_train_loss)
            
            # Validation
            if use_validation:
                feature_encoder.eval()
                val_loss = 0.0
                all_val_labels = []
                all_val_preds = []
                
                with torch.no_grad():
                    for batch_embs, batch_tracks, batch_labels in val_loader:
                        batch_embs = batch_embs.to(device)
                        batch_tracks = batch_tracks.to(device)
                        batch_labels = batch_labels.to(device)
                        
                        outputs = feature_encoder(batch_embs, batch_tracks)
                        loss = criterion(outputs, batch_labels)
                        val_loss += loss.item()
                        
                        probs = torch.softmax(outputs, dim=1)
                        preds = torch.argmax(probs, dim=1)
                        
                        all_val_labels.extend(batch_labels.cpu().numpy())
                        all_val_preds.extend(preds.cpu().numpy())
                
                avg_val_loss = val_loss / len(val_loader)
                metrics['val_loss'].append(avg_val_loss)
                
                val_acc = accuracy_score(all_val_labels, all_val_preds)
                val_f1 = f1_score(all_val_labels, all_val_preds, average=metric_avg, zero_division=0)
                metrics['val_accuracy'].append(val_acc)
                metrics['val_f1'].append(val_f1)
                
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = copy.deepcopy(feature_encoder.state_dict())
            else:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}")
                best_model_state = copy.deepcopy(feature_encoder.state_dict())
        
        # Load best model state
        if best_model_state:
            feature_encoder.load_state_dict(best_model_state)
        
        mlp_model = None
        
        # Train decoupled MLP if requested
        if decouple_mlp:
            print("\nTraining decoupled MLP classifier...")
            feature_encoder.eval()
            
            all_train_features = []
            with torch.no_grad():
                for batch_embs, batch_tracks, _ in train_loader:
                    batch_embs = batch_embs.to(device)
                    batch_tracks = batch_tracks.to(device)
                    
                    if hasattr(feature_encoder, 'extract_features'):
                        batch_features = feature_encoder.extract_features(batch_embs, batch_tracks)
                        batch_features = batch_features.cpu().numpy()
                    else:
                        # Fallback: manually extract features
                        _, (hidden_state, _) = feature_encoder.lstm(batch_tracks)
                        track_summary = hidden_state[-1]
                        combined_features = torch.cat((batch_embs, track_summary), dim=1)
                        batch_features = combined_features.cpu().numpy()
                    
                    all_train_features.append(batch_features)
            
            X_train_mlp = np.concatenate(all_train_features, axis=0)
            
            # Train sklearn MLP
            sklearn_mlp = MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                activation='relu',
                max_iter=10000,
                random_state=seed
            )
            sklearn_mlp.fit(X_train_mlp, y_train)
            
            y_train_pred_mlp = sklearn_mlp.predict(X_train_mlp)
            train_acc_mlp = accuracy_score(y_train, y_train_pred_mlp)
            metrics['mlp_train_accuracy'] = train_acc_mlp
            print(f"MLP Training Accuracy: {train_acc_mlp:.4f}")
            
            # Convert to PyTorch MLP
            input_dim = X_train_mlp.shape[1]
            mlp_model = sklearn_to_pytorch_mlp(sklearn_mlp, input_dim, num_classes, device)
            print("MLP conversion complete.")
        
        # Create IATPredictor instance
        predictor = cls(
            feature_encoder=feature_encoder,
            mlp_classifier=mlp_model,
            label_encoder=label_encoder,
            device=device
        )
        
        return predictor, metrics


