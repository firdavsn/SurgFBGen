"""
SurgFBGen: Surgical Feedback Generation Pipeline
================================================

This module provides a comprehensive pipeline for generating and evaluating surgical feedback
based on Instrument-Action-Tissue (IAT) predictions from surgical videos.

The pipeline consists of the following main components:
1. Feature Extraction: Extract visual, textual, and motion embeddings from surgical data
2. IAT Prediction: Train and evaluate predictors for instrument, action, and tissue recognition
3. Feedback Generation: Generate natural language feedback based on IAT predictions
4. Feedback Evaluation: Assess the quality of generated feedback against ground truth

Key Features:
- Multi-modal feature extraction (vision, text, motion)
- Support for multiple surgical vision-language models (SurgVLP, HecVL, PeskaVLP)
- Flexible IAT prediction with various input combinations
- LLM-based feedback generation and evaluation
- Comprehensive metrics and evaluation framework
"""

import os
import pickle
from enum import Enum
from typing import Dict, Optional

import torch
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import f1_score, roc_auc_score
from mmengine.config import Config

import surgvlp

from surgfbgen.scripts.extract_fbk_vis_embs import (
    surgvlp_embeddings,
    hecvl_embeddings,
    peskavlp_embeddings,
    SurgVLP_path,
)
from surgfbgen.models.utils import surgvlp_load
import surgfbgen.scripts.track_instruments as track_instruments
from surgfbgen.scripts.iat_predictor_train_and_eval import (
    run_via_hybrid,
    add_vis_embs,
    filter_none,
    get_embeddings_and_labels,
    get_embeddings_and_labels_multiple,
    standardize_track_shape,
)
from surgfbgen.models.iat_predictor import (
    LSTMFusionModel,
    IATPredictor,
)
from surgfbgen.models.feedback_generator import (
    FeedbackGenerator, 
    FeedbackGeneratorConfig
)
from surgfbgen.models.feedback_evaluator import (
    FeedbackEvaluator, 
    FeedbackEvaluatorConfig
)


# ==========================================
# Enumerations and Constants
# ==========================================

class VisionModelName(Enum):
    """Supported vision-language models for visual feature extraction."""
    SURGVLP = 'SurgVLP'
    HECVL = 'HecVL'
    PESKAVLP = 'PeskaVLP'


class TextModelName(Enum):
    """Supported models for text embedding extraction."""
    SURGVLP = 'SurgVLP'
    HECVL = 'HecVL'
    PESKAVLP = 'PeskaVLP'


class TextType(Enum):
    """Types of text inputs for embedding extraction."""
    PROCEDURE = 'procedure'
    TASK = 'task'


class IATColumn(Enum):
    """Instrument-Action-Tissue prediction targets."""
    INSTRUMENT = 'instrument'
    ACTION = 'action'
    TISSUE = 'tissue'


class SurgicalVLModel(Enum):
    """Surgical vision-language models for IAT prediction."""
    SURGVLP = 'SurgVLP'
    HECVL = 'HecVL'
    PESKAVLP = 'PeskaVLP'


class IATInputs(Enum):
    """Input modality combinations for IAT prediction."""
    VISION = 'vision'
    VISION_PROCEDURE = 'vision+procedure'
    VISION_TASK = 'vision+task'
    VISION_PROCEDURE_TASK = 'vision+procedure+task'
    ALL = 'vision+procedure+task'  # Alias for VISION_PROCEDURE_TASK


# ==========================================
# Feature Extraction Functions
# ==========================================

def extract_vis_embs(
    vision_model: VisionModelName,
    clips_data_dir: str,
    output_embeddings_dir: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    clip_file_type: str = 'avi',
    overwrite: bool = True,
) -> None:
    """
    Extract visual embeddings from surgical video clips using specified vision model.
    
    This function processes video clips and extracts frame-level visual features
    using pre-trained surgical vision-language models.
    
    Args:
        vision_model: The vision-language model to use for extraction.
        clips_data_dir: Directory containing video clips to process.
        output_embeddings_dir: Directory to save extracted embeddings.
        device: Computation device ('cuda' or 'cpu').
        clip_file_type: File extension of video clips (default: 'avi').
        overwrite: Whether to overwrite existing embeddings.
        
    Returns:
        None. Saves embeddings to an H5 file in the output directory.
        
    Raises:
        ValueError: If an invalid vision model is specified.
    """
    output_h5_path = os.path.join(
        output_embeddings_dir, 
        'vision', 
        f"{vision_model.value.lower()}_fbk_vis_embs.h5"
    )
    os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)
    
    if vision_model == VisionModelName.SURGVLP:
        surgvlp_embeddings(
            clips_data_dir, 
            output_h5_path, 
            device, 
            clip_file_type=clip_file_type, 
            overwrite=overwrite
        )
    elif vision_model == VisionModelName.HECVL:
        hecvl_embeddings(
            clips_data_dir, 
            output_h5_path, 
            device, 
            clip_file_type=clip_file_type, 
            overwrite=overwrite
        )
    elif vision_model == VisionModelName.PESKAVLP:
        peskavlp_embeddings(
            clips_data_dir, 
            output_h5_path, 
            device, 
            clip_file_type=clip_file_type, 
            overwrite=overwrite
        )
    else:
        raise ValueError(f"Invalid vision model: {vision_model}")
    
    print(f"Extracted embeddings for {vision_model} and saved to {output_h5_path}")


def extract_text_embs(
    text_model: TextModelName,
    text_type: TextType,
    input_csv_path: str,
    output_embeddings_dir: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> None:
    """
    Extract text embeddings from procedure or task descriptions.
    
    This function processes textual descriptions of surgical procedures or tasks
    and generates semantic embeddings using pre-trained language models.
    
    Args:
        text_model: The text model to use for embedding extraction.
        text_type: Type of text to process (PROCEDURE or TASK).
        input_csv_path: Path to CSV file containing text descriptions.
        output_embeddings_dir: Directory to save extracted embeddings.
        device: Computation device ('cuda' or 'cpu').
        
    Returns:
        None. Saves embeddings to a Parquet file in the output directory.
        
    Raises:
        ValueError: If an invalid text model or text type is specified.
        
    Notes:
        The input CSV must contain appropriate columns:
        - 'procedure_defn' for PROCEDURE type
        - 'task_defn' for TASK type
    """
    # Model configuration selection
    config_mapping = {
        TextModelName.SURGVLP: 'config_surgvlp.py',
        TextModelName.HECVL: 'config_hecvl.py',
        TextModelName.PESKAVLP: 'config_peskavlp.py',
    }
    
    if text_model not in config_mapping:
        raise ValueError(f"Invalid text model: {text_model}")
    
    config_file = config_mapping[text_model]
    configs = Config.fromfile(
        os.path.join(SurgVLP_path, 'tests', config_file), 
        lazy_import=False
    )['config']
    
    # Load model and tokenizer
    model, tokenizer = surgvlp_load(
        configs.model_config, 
        device=device, 
        strict_load_state_dict=False
    )
    
    # Determine text column based on type
    text_col_mapping = {
        TextType.PROCEDURE: 'procedure_defn',
        TextType.TASK: 'task_defn',
    }
    
    if text_type not in text_col_mapping:
        raise ValueError(f"Invalid text type: {text_type}")
    
    text_col = text_col_mapping[text_type]
    
    # Load and process text data
    df = pd.read_csv(input_csv_path)
    text_embs = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        text = surgvlp.tokenize(row[text_col], device=device)
        
        with torch.no_grad():
            output = model(None, text, mode='text')
            text_emb = output['text_emb']
            text_emb = text_emb.cpu().numpy()
            text_emb = text_emb.squeeze().tolist()
            text_embs.append([float(e) for e in text_emb])
    
    # Save embeddings
    df[f"{text_col}_emb-{text_model.value}"] = text_embs
    
    output_parquet_path = os.path.join(
        output_embeddings_dir,
        "text",
        f"{text_model.value.lower()}_{text_type.value.lower()}_embs.parquet"
    )
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    df.to_parquet(output_parquet_path)
    
    print(f"Extracted text embeddings from {input_csv_path} via {text_model} "
          f"and saved to {output_parquet_path}")

def extract_motion_embeddings(
    clips_dir: str,
    output_h5_path: str,
    cotracker_checkpoint_path: str,
    overwrite: bool = False,
    filter_instrument: bool = True,
    filter_instrument_num_tracks: int = 15
) -> None:
    """
    Extract motion embeddings from surgical video clips using CoTracker.
    
    This function tracks instrument movements across video frames to capture
    motion patterns that are crucial for understanding surgical actions.
    The tracking data can be filtered to focus on instrument-specific movements,
    which improves the quality of downstream IAT predictions.
    
    Args:
        clips_dir: Directory containing surgical video clips to process.
            Expected to contain video files (e.g., .avi, .mp4) organized
            by clip ID.
        output_h5_path: Path where the motion tracking data will be saved
            in HDF5 format. The file will contain tracked points for each
            video clip indexed by clip ID.
        cotracker_checkpoint_path: Path to the pre-trained CoTracker model
            checkpoint file. This should be a .pth file containing the
            model weights.
        overwrite: If True, overwrites existing output file if it exists.
            If False and file exists, skips processing (default: False).
        filter_instrument: If True, applies filtering to focus on instrument
            movements and removes background motion. This typically improves
            prediction accuracy (default: True).
        filter_instrument_num_tracks: Number of tracking points to retain
            after filtering. Only used if filter_instrument is True.
            Higher values capture more detail but increase computational
            cost (default: 15).
    
    Returns:
        None. Saves motion tracking data to the specified H5 file.
        
    Raises:
        FileNotFoundError: If clips_dir or cotracker_checkpoint_path doesn't exist.
        IOError: If unable to write to output_h5_path.
        
    Notes:
        - The output H5 file structure will be:
            {clip_id: array of shape (1, num_frames, num_tracks, 2)}
          where the last dimension contains (x, y) coordinates
        - CoTracker is a state-of-the-art point tracking model that can
          track multiple points across video frames
        - Instrument filtering uses motion patterns to identify and retain
          tracks that likely correspond to surgical instruments
        - Processing time depends on video resolution and length; expect
          several seconds per clip on GPU
        
    """
    track_instruments.run(
        clips_dir=clips_dir,
        output_h5_path=output_h5_path,
        cotracker_checkpoint_path=cotracker_checkpoint_path,
        overwrite=overwrite,
        filter_instrument=filter_instrument,
        filter_instrument_num_tracks=filter_instrument_num_tracks
    )

# ==========================================
# IAT Predictor Training and Evaluation
# ==========================================

def run_iat_predictor_train_and_eval(
    iat_col: IATColumn,
    model: SurgicalVLModel,
    inputs: IATInputs,
    output_json_path: str,
    annotations_path: str,
    embeddings_dir: str,
    pred_csv_path: Optional[str] = None,
    num_none_included: int = 100,
    seed: int = 0,
    num_folds: int = 5,
    num_tracks: int = 15,
    metric_avg: str = 'macro',
    multiple_instance_training: bool = False,
) -> Dict:
    """
    Train and evaluate an IAT predictor using cross-validation.
    
    This function performs k-fold cross-validation to train and evaluate
    an IAT predictor for a specific target (instrument, action, or tissue).
    
    Args:
        iat_col: The IAT column to predict (INSTRUMENT, ACTION, or TISSUE).
        model: The surgical vision-language model to use.
        inputs: Input modality combination for prediction.
        output_json_path: Path to save evaluation metrics in JSON format.
        annotations_path: Path to CSV file containing IAT annotations.
        embeddings_dir: Directory containing pre-extracted embeddings.
        pred_csv_path: Optional path to save predictions CSV.
        num_none_included: Number of 'None' samples to include in training.
        seed: Random seed for reproducibility.
        num_folds: Number of folds for cross-validation.
        num_tracks: Number of motion tracks to use.
        metric_avg: Averaging method for metrics ('macro', 'weighted', 'micro').
        multiple_instance_training: Whether to use multiple instance learning.
        
    Returns:
        Dictionary containing evaluation metrics and results.
        
    Notes:
        This function expects the embeddings directory to have the following structure:
        - embeddings_dir/
            - vision/: Visual embeddings
            - text/: Text embeddings (procedures and tasks)
            - motion/: Motion tracking data
        
        The function expects annotations_path to point to a DF with columns:
            dialogue,timestamp,case,cvid,procedure,procedure_defn,task,task_defn,instrument,action,tissue,instrument-extraction,action-extraction,tissue-extraction
    """
    # Construct paths to embedding files
    vision_embeddings_dir = os.path.join(embeddings_dir, 'vision')
    procedures_embs_path = os.path.join(
        embeddings_dir, 
        'text', 
        f'{model.value.lower()}_procedure_embs.parquet'
    )
    tasks_embs_path = os.path.join(
        embeddings_dir, 
        'text', 
        f'{model.value.lower()}_task_embs.parquet'
    )
    instrument_tracks_path = os.path.join(
        embeddings_dir, 
        'motion', 
        f'instrument_tracks-num_tracks={num_tracks}.h5'
    )
    
    # Delegate to the existing implementation
    return run_via_hybrid(
        iat_col=iat_col.value.lower(),
        model=model.value.lower(),
        inputs=inputs.value,
        output_json=output_json_path,
        pred_csv=pred_csv_path,
        num_none_included=num_none_included,
        num_folds=num_folds,
        vision_embeddings_dir=vision_embeddings_dir,
        annotations_path=annotations_path,
        procedures_embs_path=procedures_embs_path,
        tasks_embs_path=tasks_embs_path,
        instrument_tracks_path=instrument_tracks_path,
        seed=seed,
        num_tracks=num_tracks,
        metric_avg=metric_avg,
        multiple_instance_training=multiple_instance_training,
    )

def train_via_hybrid(
    iat_col: str,
    model: str,
    inputs: str,
    model_save_path: str,
    num_none_included: int = 100,
    train_test_split_ratio: float = 0.8,
    use_validation_set: bool = True,
    vision_embeddings_dir: str = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/outputs/embeddings/vision',
    annotations_path: str = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/data/iat_predictor_splits/full.csv',
    procedures_embs_path: str = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/data/iat/procedures_embs_df.parquet',
    tasks_embs_path: str = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/data/iat/tasks_embs_df.parquet',
    instrument_tracks_path: str = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/outputs/instrument_tracks/instrument_tracks.h5',
    seed: int = 42,
    num_tracks: int = 15,
    multiple_instance_training: bool = False,
    decouple_mlp: bool = False,
    metric_avg: str = 'weighted',
    epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    lstm_hidden_dim: int = 32,
    num_lstm_layers: int = 10,
    dropout: float = 0.2,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    """
    Trains a hybrid model for IAT prediction.
    
    This function handles data loading and preparation, then delegates the actual
    training to IATPredictor.train_model().
    
    Args:
        iat_col (str): The IAT column to predict.
        model (str): Model name for vision embeddings.
        inputs (str): Input type ('vision', 'vision+procedure', 'vision+task', 'vision+procedure+task').
        model_save_path (str): Path to save the trained model.
        num_none_included (int): Number of 'None' samples to include.
        train_test_split_ratio (float): Ratio for train/validation split.
        use_validation_set (bool): Whether to use a validation set.
        vision_embeddings_dir (str): Directory containing vision embeddings.
        annotations_path (str): Path to annotations CSV.
        procedures_embs_path (str): Path to procedure embeddings.
        tasks_embs_path (str): Path to task embeddings.
        instrument_tracks_path (str): Path to instrument tracks H5 file.
        seed (int): Random seed.
        num_tracks (int): Number of tracks to use.
        multiple_instance_training (bool): Whether to use multiple instance training.
        decouple_mlp (bool): Whether to train a separate MLP classifier.
        metric_avg (str): Averaging method for metrics.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate.
        lstm_hidden_dim (int): LSTM hidden dimension.
        num_lstm_layers (int): Number of LSTM layers.
        dropout (float): Dropout rate.
        device (str): Device to use ('cuda' or 'cpu').
    
    Returns:
        IATPredictor: A trained predictor model.
        
    Notes:
        The function expects annotations_path to point to a DF with columns:
            dialogue,timestamp,case,cvid,procedure,procedure_defn,task,task_defn,instrument,action,tissue,instrument-extraction,action-extraction,tissue-extraction
    """
    print(f"Training IAT Predictor")
    print(f"IAT Column: {iat_col}")
    print(f"Model: {model}")
    print(f"Inputs: {inputs} + tracks")
    print(f"Device: {device}")
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    # Load and process data
    print("\nLoading and processing data...")
    annotations_df = pd.read_csv(annotations_path).replace('NONE', None)
    annotations_df = add_vis_embs(model, annotations_df, vision_embeddings_dir, force=True)
    annotations_df = annotations_df.dropna(subset=[f'{model}_vis_embs']).copy()
    annotations_df = filter_none(annotations_df, num_none_included, iat_col, seed=seed)
    
    # Get embeddings based on input type
    if inputs == 'vision':
        if not multiple_instance_training:
            processed_df = get_embeddings_and_labels(annotations_df, None, None, iat_col, model)
        else:
            processed_df = get_embeddings_and_labels_multiple(
                annotations_df, None, None, iat_col, model, current_fps=5, target_fps=1
            )
    elif inputs == 'vision+procedure':
        procedures_df = pd.read_parquet(procedures_embs_path)
        if not multiple_instance_training:
            processed_df = get_embeddings_and_labels(annotations_df, procedures_df, None, iat_col, model)
        else:
            processed_df = get_embeddings_and_labels_multiple(
                annotations_df, procedures_df, None, iat_col, model, current_fps=5, target_fps=1
            )
    elif inputs == 'vision+task':
        tasks_df = pd.read_parquet(tasks_embs_path)
        if not multiple_instance_training:
            processed_df = get_embeddings_and_labels(annotations_df, None, tasks_df, iat_col, model)
        else:
            processed_df = get_embeddings_and_labels_multiple(
                annotations_df, None, tasks_df, iat_col, model, current_fps=5, target_fps=1
            )
    elif inputs == 'vision+procedure+task':
        procedures_df = pd.read_parquet(procedures_embs_path)
        tasks_df = pd.read_parquet(tasks_embs_path)
        if not multiple_instance_training:
            processed_df = get_embeddings_and_labels(annotations_df, procedures_df, tasks_df, iat_col, model)
        else:
            processed_df = get_embeddings_and_labels_multiple(
                annotations_df, procedures_df, tasks_df, iat_col, model, current_fps=5, target_fps=1
            )
    else:
        raise ValueError(f"Inputs type {inputs} is not supported.")
    
    # Load instrument tracks
    print("Loading instrument tracks...")
    h5 = h5py.File(instrument_tracks_path, 'r')
    processed_df['tracks'] = processed_df['cvid'].apply(
        lambda x: standardize_track_shape(h5[x][:][0], target_dots=num_tracks) if x in h5 else None
    )
    h5.close()
    
    # Clean data
    processed_df.dropna(subset=['tracks', 'embedding'], inplace=True)
    processed_df = processed_df.reset_index(drop=True)
    print(f"Number of samples: {len(processed_df)}")
    
    # Prepare arrays
    embeddings = np.array(processed_df['embedding'].values.tolist())
    tracks = np.array(processed_df['tracks'].values.tolist())
    labels = processed_df['label'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {[str(c) for c in label_encoder.classes_]}")
    
    # Reshape tracks
    tracks_reshaped = tracks.reshape(tracks.shape[0], tracks.shape[1], -1)
    
    # Train/validation split
    if train_test_split_ratio < 1.0 and use_validation_set:
        indices = np.arange(len(processed_df))
        train_indices, val_indices = train_test_split(
            indices,
            train_size=train_test_split_ratio,
            random_state=seed,
            stratify=encoded_labels
        )
        
        X_train_embs = embeddings[train_indices]
        X_val_embs = embeddings[val_indices]
        X_train_tracks = tracks_reshaped[train_indices]
        X_val_tracks = tracks_reshaped[val_indices]
        y_train = encoded_labels[train_indices]
        y_val = encoded_labels[val_indices]
        
        print(f"Training samples: {len(train_indices)}, Validation samples: {len(val_indices)}")
    else:
        X_train_embs = embeddings
        X_train_tracks = tracks_reshaped
        y_train = encoded_labels
        X_val_embs, X_val_tracks, y_val = None, None, None
        print("Using entire dataset for training.")
    
    # Train the model using IATPredictor's class method
    predictor, metrics = IATPredictor.train_model(
        X_train_embs=X_train_embs,
        X_train_tracks=X_train_tracks,
        y_train=y_train,
        X_val_embs=X_val_embs,
        X_val_tracks=X_val_tracks,
        y_val=y_val,
        feature_encoder_class=LSTMFusionModel,  # You'll need to import this
        num_classes=num_classes,
        label_encoder=label_encoder,
        decouple_mlp=decouple_mlp,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lstm_hidden_dim=lstm_hidden_dim,
        num_lstm_layers=num_lstm_layers,
        dropout=dropout,
        metric_avg=metric_avg,
        seed=seed,
        device=device
    )
    
    # Save the model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    predictor.save(model_save_path)
    print(f"\nSaved IATPredictor to {model_save_path}")
    
    # Attach additional metadata for convenience
    predictor.training_metrics = metrics
    predictor.processed_df = processed_df
    
    return predictor

# =========================================
# IAT Prediction and Evaluation
# =========================================

def run_predictions(
    predictor,
    data_df: pd.DataFrame,
    verbose: bool = True,
    metric_avg: str = 'weighted'
) -> pd.DataFrame:
    """
    Run predictions on provided dataframe and calculate AUROC and F1 score.
    
    Args:
        predictor: Trained IATPredictor instance
        data_df: DataFrame containing 'embedding', 'tracks', 'cvid', and 'label' columns
        verbose: Whether to print detailed results
        metric_avg: Averaging method for F1 and AUROC ('weighted', 'macro', 'micro')
        
    Returns:
        DataFrame with predictions and results
    """
    if data_df is None or len(data_df) == 0:
        raise ValueError("data_df must be provided and non-empty")
    
    if not all(col in data_df.columns for col in ['embedding', 'tracks', 'cvid', 'label']):
        raise ValueError("data_df must contain columns: 'embedding', 'tracks', 'cvid', 'label'")
    
    results = []
    
    if verbose:
        print(f"\nMaking predictions on {len(data_df)} samples:")
        print("-" * 40)
    
    all_true_labels = []
    all_pred_labels = []
    all_pred_probs = []
    
    for idx, (_, row) in enumerate(data_df.iterrows()):
        # Get embedding and track for this sample
        embedding = np.array(row['embedding'])
        track = np.array(row['tracks'])
        
        # ... (Keep the existing track reshaping logic) ...
        if len(track.shape) == 2:
            num_points = track.shape[0]
            track_flat = track.flatten()
            track_reshaped = np.tile(track_flat, (num_points, 1))
        else:
            track_reshaped = track.reshape(track.shape[0], -1)
        
        # Make prediction with reshaped track
        predicted_class_idx = predictor.predict(embedding, track_reshaped, return_probs=False)[0]
        predicted_label = predictor.predict_labels(embedding, track_reshaped)[0]
        
        # Get probabilities for all classes
        probs = predictor.predict(embedding, track_reshaped, return_probs=True)[0]
        
        # Store results
        sample_result = {
            'cvid': row['cvid'],
            'true_label': row['label'],
            'predicted_label': predicted_label,
            'confidence': probs[predicted_class_idx],
            'correct': row['label'] == predicted_label # Keep for reference
        }
        
        # Add probability for each class
        for i, class_name in enumerate(predictor.label_encoder.classes_):
            sample_result[f'prob_{class_name}'] = probs[i]
        
        results.append(sample_result)
        
        # Store for overall metrics calculation
        all_true_labels.append(row['label'])
        all_pred_labels.append(predicted_label)
        all_pred_probs.append(probs)
        
        # ... (Keep the existing verbose printing for individual samples) ...
        if verbose:
            print(f"\nSample {idx + 1}:")
            print(f"  CVID: {row['cvid']}")
            print(f"  True Label: {row['label']}")
            print(f"  Predicted Label: {predicted_label}")
            print(f"  Confidence: {probs[predicted_class_idx]:.3f}")
            if len(probs) > 1:
                top_3_indices = np.argsort(probs)[::-1][:3]
                print(f"  Top 3 Predictions:")
                for rank, class_idx in enumerate(top_3_indices, 1):
                    class_label = predictor.label_encoder.inverse_transform([class_idx])[0]
                    print(f"    {rank}. {class_label}: {probs[class_idx]:.3f}")

    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    if verbose and len(results_df) > 0:
        # --- Calculate and Print Summary Statistics (AUROC and F1) ---
        print("\n" + "-" * 40)
        print("Summary:")
        
        all_pred_probs = np.array(all_pred_probs)
        all_classes = predictor.label_encoder.classes_
        
        # Binarize labels for AUROC
        # Ensure labels used for binarizing match the ones in all_true_labels
        unique_labels_in_data = sorted(list(set(all_true_labels)))
        # Use all_classes from the encoder to ensure consistent ordering
        all_true_binarized = label_binarize(all_true_labels, classes=all_classes)
        
        # Calculate F1 Score
        f1 = f1_score(all_true_labels, all_pred_labels, average=metric_avg, zero_division=0)
        print(f"  Overall F1 Score ({metric_avg}): {f1:.4f}")

        # Calculate AUROC
        # Handle different numbers of classes
        if len(all_classes) == 2:
            # Binary case: use probabilities of the positive class
            # Assuming the second class is the positive one
            auroc = roc_auc_score(all_true_labels, all_pred_probs[:, 1])
            print(f"  Overall AUROC: {auroc:.4f}")
        elif len(all_classes) > 2:
            # Multi-class case
            try:
                auroc = roc_auc_score(all_true_binarized, all_pred_probs, average=metric_avg, multi_class='ovr')
                print(f"  Overall AUROC ({metric_avg}, OVR): {auroc:.4f}")
            except ValueError as e:
                print(f"  Could not calculate AUROC: {e}")
        else:
            print("  AUROC calculation skipped (only one class present).")
            
    return results_df


def run_all_predictions_and_save(
    predictors: Dict[str, IATPredictor],
    output_path: str,
    verbose: bool = True,
    metric_avg: str = 'weighted'
) -> pd.DataFrame:
    """
    Runs predictions for each IAT predictor (instrument, action, tissue),
    combines them into a single DataFrame, and saves to a CSV.
    
    The output CSV will contain renamed columns for each IAT type, e.g.,
    'instrument_pred', 'instrument_conf', 'instrument_pred_probs', etc.
    
    Args:
        predictors: Dictionary mapping IAT column name (str) to a trained IATPredictor instance.
        output_path: Path to save the combined CSV file.
        verbose: Whether to print summary statistics for each predictor.
        metric_avg: Averaging method for F1 and AUROC ('weighted', 'macro', 'micro').
        
    Returns:
        DataFrame with all combined predictions and metrics.
    """
    combined_df = None
    
    # Define common columns to merge on. Assumes all predictors were trained
    # with data containing these base columns.
    # We find the smallest common set of non-prediction columns from the first predictor.
    # This is safer than hardcoding.
    
    first_predictor = next(iter(predictors.values()))
    if not hasattr(first_predictor, 'processed_df'):
         raise ValueError("Predictor has no 'processed_df'. Cannot determine merge columns.")
         
    base_cols = [
        col for col in first_predictor.processed_df.columns 
        if col not in ['embedding', 'tracks', 'label', 'pred', 'confidence', 'correct', 'pred_probs'] 
        and not col.startswith('prob_')
    ]
    # Ensure 'cvid' is present, as it's the primary key
    if 'cvid' not in base_cols:
        base_cols.insert(0, 'cvid')
        
    # Also include the original true labels
    base_cols.extend(['instrument', 'action', 'tissue'])
    # Remove duplicates
    base_cols = sorted(list(set(base_cols))) 
    
    # In case 'processed_df' wasn't stored, fall back to a minimal set
    if 'cvid' not in base_cols:
        base_cols = ['cvid'] # Failsafe

    if verbose:
        print(f"Combining predictions. Merging on columns: {base_cols}")

    for iat_col, predictor in predictors.items():
        if verbose:
            print(f"\n{'-' * 40}\nGenerating predictions for: {iat_col}\n{'-' * 40}")
            
        # Use the predictor's own processed_df as the data source
        if not hasattr(predictor, 'processed_df'):
            raise ValueError(f"Predictor for '{iat_col}' has no 'processed_df' attached.")
        
        data_df = predictor.processed_df.copy()

        # --- Run Batch Predictions ---
        all_embeddings = np.array(data_df['embedding'].values.tolist())
        all_tracks = np.array(data_df['tracks'].values.tolist())
        
        if len(all_tracks.shape) == 3:
            n_samples, seq_len, coords = all_tracks.shape
            if coords == 2:
                all_tracks_reshaped = []
                for track in all_tracks:
                    track_flat = track.flatten()
                    track_repeated = np.tile(track_flat, (seq_len, 1))
                    all_tracks_reshaped.append(track_repeated)
                all_tracks = np.array(all_tracks_reshaped)
            else:
                pass
        else:
            all_tracks = all_tracks.reshape(all_tracks.shape[0], all_tracks.shape[1], -1)
        
        all_predictions = predictor.predict_labels(all_embeddings, all_tracks)
        all_probs = predictor.predict(all_embeddings, all_tracks, return_probs=True)
        all_classes = predictor.label_encoder.classes_
        
        # --- Create results dataframe for this IAT column ---
        results_df = data_df[base_cols].copy()
        results_df = results_df.drop(columns=[col for col in results_df.columns if 'embs' in col])

        # Add prediction columns
        results_df[f'{iat_col}_pred'] = all_predictions
        results_df[f'{iat_col}_conf'] = np.max(all_probs, axis=1)
        
        # Add pred_probs as a string dictionary
        pred_probs_list = []
        for i in range(len(all_probs)):
            prob_dict = {str(all_classes[j]): float(all_probs[i][j]) for j in range(len(all_classes))}
            pred_probs_list.append(str(prob_dict))
        results_df[f'{iat_col}_pred_probs'] = pred_probs_list
        
        # --- Optional: Add true label and 'correct' for this specific task ---
        # 'label' in data_df is specific to this predictor
        results_df[f'{iat_col}_true'] = data_df['label'] 
        results_df[f'{iat_col}_correct'] = results_df[f'{iat_col}_true'] == results_df[f'{iat_col}_pred']

        # --- Merge with combined_df ---
        if combined_df is None:
            combined_df = results_df
        else:
            # Merge results, dropping duplicated base columns from the new df
            merge_cols = [col for col in results_df.columns if col in base_cols]
            for col in combined_df.columns:
                if isinstance(combined_df.iloc[0][col], np.ndarray):
                    print(f"col = {col}, combined_df.iloc[0][col] = {combined_df.iloc[0][col]}")
            for col in results_df.columns:
                if isinstance(results_df.iloc[0][col], np.ndarray):
                    print(f"col = {col}, results_df.iloc[0][col] = {results_df.iloc[0][col]}")
            combined_df = combined_df.merge(
                results_df, 
                on=merge_cols, 
                how='outer'
            )
        
        if verbose:
            # --- Calculate and Print Summary Statistics (AUROC and F1) ---
            print(f"\nOverall Statistics for {iat_col} (avg='{metric_avg}'):")
            print(f"  Total Samples: {len(results_df)}")
            
            true_labels = results_df[f'{iat_col}_true']
            pred_labels = results_df[f'{iat_col}_pred']
            
            all_true_binarized = label_binarize(true_labels, classes=all_classes)
            
            f1 = f1_score(true_labels, pred_labels, average=metric_avg, zero_division=0)
            print(f"  Overall F1 Score: {f1:.4f}")

            if len(all_classes) == 2:
                auroc = roc_auc_score(true_labels, all_probs[:, 1])
                print(f"  Overall AUROC: {auroc:.4f}")
            elif len(all_classes) > 2:
                try:
                    auroc = roc_auc_score(all_true_binarized, all_probs, average=metric_avg, multi_class='ovr')
                    print(f"  Overall AUROC (OVR): {auroc:.4f}")
                except ValueError as e:
                    print(f"  Could not calculate AUROC: {e}")
            else:
                print("  AUROC calculation skipped (only one class present).")
            
            overall_accuracy = results_df[f'{iat_col}_correct'].mean()
            print(f"  Overall Accuracy: {overall_accuracy:.4f}")

    # --- Save Combined DataFrame ---
    final_output_path = os.path.expanduser(output_path)
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    combined_df.to_csv(final_output_path, index=False)
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"Saved COMBINED predictions to: {final_output_path}")
        print(f"Combined DataFrame shape: {combined_df.shape}")
        print("=" * 60)
    
    return combined_df

# =========================================
# Feedback Generation
# =========================================

def generate_feedback(
    predictions_csv_path: str,
    output_csv_path: str,
    clips_dir: str,
    config_params: dict,
    iat_instrument_col: str = 'instrument_pred',
    iat_action_col: str = 'action_pred',
    iat_tissue_col: str = 'tissue_pred',
    api_key_env_var: str = 'OPENAI_API_KEY',
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generates feedback for IAT predictions using a FeedbackGenerator.

    Args:
        predictions_csv_path (str): Path to the CSV file containing IAT predictions.
        output_csv_path (str): Path to save the final DataFrame with feedback.
        clips_dir (str): Directory containing video clips (for frame extraction if enabled).
        config_params (dict): Dictionary of parameters for FeedbackGeneratorConfig.
        iat_instrument_col (str): Column name for instrument predictions.
        iat_action_col (str): Column name for action predictions.
        iat_tissue_col (str): Column name for tissue predictions.
        api_key_env_var (str): Environment variable name for the LLM API key.
        verbose (bool): Whether to print status messages.

    Returns:
        pd.DataFrame: The DataFrame with the generated feedback column.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Generating Feedback")
        print("=" * 60)
        print(f"Loading predictions from: {predictions_csv_path}")

    # Load and filter predictions
    # Use os.path.expanduser to handle '~' in paths
    full_split_df = pd.read_csv(os.path.expanduser(predictions_csv_path))
    full_split_df = full_split_df.fillna('NONE')
    
    if verbose:
        print(f"Loaded {len(full_split_df)} predictions.")
    
    # Filter out rows that are all UNCERTAIN or all NONE
    full_split_df = full_split_df[
        (full_split_df[iat_instrument_col] != 'UNCERTAIN') |
        (full_split_df[iat_action_col] != 'UNCERTAIN') |
        (full_split_df[iat_tissue_col] != 'UNCERTAIN')
    ]
    full_split_df = full_split_df[
        (full_split_df[iat_instrument_col] != 'NONE') |
        (full_split_df[iat_action_col] != 'NONE') |
        (full_split_df[iat_tissue_col] != 'NONE')
    ]
    
    if verbose:
        print(f"Filtered to {len(full_split_df)} samples for feedback generation.")
        print(f"Using IAT columns: {iat_instrument_col}, {iat_action_col}, {iat_tissue_col}")

    # Initialize config
    config = FeedbackGeneratorConfig(
        clips_dir=os.path.expanduser(clips_dir),
        **config_params
    )
    if verbose:
        print(f"FeedbackGenerator Config:\n{config}")

    # Get API key
    api_key = os.environ.get(api_key_env_var)
    if not api_key:
        raise ValueError(f"API key not found. Please set the {api_key_env_var} environment variable.")

    # Initialize generator
    feedback_generator = FeedbackGenerator(
        config=config,
        api_key=api_key,
        instrument_col=iat_instrument_col,
        action_col=iat_action_col,
        tissue_col=iat_tissue_col,
    )

    # Generate feedback
    if verbose:
        print("Starting feedback generation (this may take a long time)...")
    feedback_df = feedback_generator.generate_all_feedback(
        full_split_df,
    )

    # Save results
    final_output_path = os.path.expanduser(output_csv_path)
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    feedback_df.to_csv(final_output_path, index=False)
    
    if verbose:
        print(f"\nFeedback generation complete.")
        print(f"Saved feedback DataFrame to: {final_output_path}")
        print("=" * 60)
        
    return feedback_df

# =========================================
# Feedback Evaluation
# =========================================

def evaluate_feedback(
    feedback_csv_path: str,
    output_csv_path: str,
    config_params: dict,
    api_key_env_var: str,
    ground_truth_col: str,
    generated_col: str,
    save_pickle: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluates generated feedback against ground truth using a FeedbackEvaluator.

    Args:
        feedback_csv_path (str): Path to the CSV file containing generated feedback.
        output_csv_path (str): Path to save the final DataFrame with scores.
        config_params (dict): Dictionary of parameters for FeedbackEvaluatorConfig.
        api_key_env_var (str): Environment variable name for the LLM API key.
        ground_truth_col (str): Column name for the ground truth feedback.
        generated_col (str): Column name for the generated feedback.
        save_pickle (bool): Whether to also save the scores as a pickle file.
        verbose (bool): Whether to print status messages.

    Returns:
        pd.DataFrame: The DataFrame with the generated scores.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Evaluating Feedback")
        print("=" * 60)
        print(f"Loading feedback from: {feedback_csv_path}")

    # Load feedback data
    feedback_df = pd.read_csv(os.path.expanduser(feedback_csv_path))

    # Initialize config
    config = FeedbackEvaluatorConfig(**config_params)
    if verbose:
        print(f"FeedbackEvaluator Config:\n{config}")

    # Get API key
    api_key = os.environ.get(api_key_env_var)
    if not api_key:
        raise ValueError(f"API key not found. Please set the {api_key_env_var} environment variable.")

    # Initialize evaluator
    feedback_evaluator = FeedbackEvaluator(
        config=config,
        api_key=api_key,
    )

    # Generate scores
    if verbose:
        print("Starting feedback evaluation (this may take a long time)...")
        print(f"Ground Truth Column: '{ground_truth_col}'")
        print(f"Generated Column: '{generated_col}'")
        
    scores_df = feedback_evaluator.generate_all_scores(
        feedback_df,
        gt_fb_col=ground_truth_col,
        gen_fb_col=generated_col
    )

    # Save results
    final_output_path = os.path.expanduser(output_csv_path)
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    scores_df.to_csv(final_output_path, index=False)
    
    if verbose:
        print(f"\nFeedback evaluation complete.")
        print(f"Saved scores DataFrame to: {final_output_path}")

    # Save pickle file
    if save_pickle:
        pickle_path = final_output_path.replace('.csv', '.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(scores_df, f)
        if verbose:
            print(f"Saved scores pickle to: {pickle_path}")
            
    if verbose:
        print("=" * 60)
        
    return scores_df
