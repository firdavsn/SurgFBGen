import pandas as pd
import json
import numpy as np
import os
import h5py
import math

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    average_precision_score, roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix as compute_confusion_matrix,
    brier_score_loss
)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, label_binarize

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, average_precision_score, confusion_matrix as compute_confusion_matrix
import numpy as np
import pandas as pd
from tqdm import tqdm
import collections
from scipy.stats import mode

def add_vis_embs(name, annotations_df, embedding_dir, force=False):
    annotations_df = annotations_df.copy()
    col_name = f'{name}_vis_embs'
    if col_name not in annotations_df.columns or force:
        embs_path = os.path.join(embedding_dir, f"{name}_fbk_vis_embs.h5")
        embs_h5 = h5py.File(embs_path, 'r')
        annotations_df[col_name] = annotations_df['cvid'].apply(lambda x: embs_h5[x][:] if x in embs_h5 else None)
        embs_h5.close()
    return annotations_df


def evaluate_via_embs(
    processed_df, 
    metric_avg='weighted', 
    sampler=None, 
    num_folds=5, 
    hidden_layer_sizes=(128, 64),
    seed=42,
):
    """
    Evaluates an MLP model on embeddings using stratified k-fold cross-validation.
    """
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    auprc_scores, auroc_scores, accuracy_scores = [], [], []
    precision_scores, recall_scores, f1_scores = [], [], []
    ece_scores = []
    confusion_matrices = []

    embeddings = np.array(processed_df['embedding'].values.tolist())
    labels = processed_df['label'].values
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    pred_df = processed_df.copy()
    # pred_df['pred'] = None
    pred_df['confidence'] = 0.0
    # pred_df['pred_probs'] = None
    # pred_df['pred_probs'] = pred_df['pred_probs'].astype('object')

    fold_num = 1
    for train_index, test_index in skf.split(embeddings, encoded_labels):
        print(f"Fold {fold_num}/{num_folds}")
        fold_num += 1
        
        X_train, X_test = embeddings[train_index], embeddings[test_index]
        y_train, y_test = encoded_labels[train_index], encoded_labels[test_index]
        
        if sampler is not None:
            X_train, y_train = sampler.fit_resample(X_train, y_train)
        
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation='relu', max_iter=1000, random_state=seed)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)
        
        pred_df.loc[test_index, 'pred'] = label_encoder.inverse_transform(y_pred)
        
        confidences = np.max(y_pred_prob, axis=1)
        pred_df.loc[test_index, 'confidence'] = confidences
        
        for i, idx in enumerate(test_index):
            probs_dict = {label_encoder.classes_[j]: float(y_pred_prob[i][j]) for j in range(num_classes)}
            pred_df.at[idx, 'pred_probs'] = str(probs_dict)
        
        # Binarize y_test for AUPRC calculation
        y_test_binarized = label_binarize(y_test, classes=np.arange(num_classes))
        
        # Ensure y_pred_prob has probabilities for all classes
        if y_test_binarized.shape[1] != y_pred_prob.shape[1]:
            full_prob = np.zeros((len(y_test), num_classes))
            model_classes_indices = [np.where(label_encoder.classes_ == c)[0][0] for c in model.classes_]
            full_prob[:, model_classes_indices] = y_pred_prob
            y_pred_prob = full_prob

        auprc_scores.append(average_precision_score(y_test_binarized, y_pred_prob, average=metric_avg))
        
        auroc_scores.append(roc_auc_score(y_test, y_pred_prob, multi_class='ovo', average=metric_avg))
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred, zero_division=0, average=metric_avg))
        recall_scores.append(recall_score(y_test, y_pred, zero_division=0, average=metric_avg))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0, average=metric_avg))
        ece_scores.append(_calculate_ece(y_test, y_pred_prob))
        confusion_matrices.append(compute_confusion_matrix(y_test, y_pred, labels=np.arange(num_classes)))
    
    pred_df = pred_df.drop(columns=[col for col in pred_df.columns if 'emb' in col])
    
    mean_auprc = float(np.mean(auprc_scores))
    mean_auroc = float(np.mean(auroc_scores))
    mean_accuracy = float(np.mean(accuracy_scores))
    mean_precision = float(np.mean(precision_scores))
    mean_recall = float(np.mean(recall_scores))
    mean_f1 = float(np.mean(f1_scores))
    mean_ece = float(np.mean(ece_scores))
    
    confusion_matrix = np.sum(confusion_matrices, axis=0)
    confusion_matrix = [list(map(int, row)) for row in confusion_matrix]
    
    metrics = {
        'auprc_scores': auprc_scores,
        'auroc_scores': auroc_scores,
        'accuracy_scores': accuracy_scores,
        'precision_scores': precision_scores,
        'recall_scores': recall_scores,
        'f1_scores': f1_scores,
        'ece_scores': ece_scores,
        'auprc_mean': mean_auprc,
        'auroc_mean': mean_auroc,
        'accuracy_mean': mean_accuracy,
        'precision_mean': mean_precision,
        'recall_mean': mean_recall,
        'f1_mean': mean_f1,
        'ece_mean': mean_ece,
        'confusion_matrix': confusion_matrix,
        'confusion_matrix_labels': label_encoder.classes_.tolist(),
    }
    
    return metrics, pred_df


class TrackDataset(Dataset):
    def __init__(self, tracks, labels):
        self.tracks = torch.tensor(tracks, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tracks[idx], self.labels[idx]

def filter_none(df, num_none_included, col, seed=42):
    if num_none_included == 'all':
        df[col] = df[col].replace(np.nan, 'None')
        df.dropna(subset=[col], inplace=True) # Ensure no NaNs remain
    elif isinstance(num_none_included, int) and num_none_included > 0:
        df_valid_labels = df.dropna(subset=[col]).copy()
        df_none_labels = df[df[col].isna()].copy()
        if len(df_none_labels) > num_none_included:
            df_none_labels = df_none_labels.sample(n=num_none_included, random_state=seed)
        df_none_labels[col] = 'None'
        df = pd.concat([df_valid_labels, df_none_labels])
    else:
        df.dropna(subset=[col], inplace=True)
    return df
    
def get_vision_embeddings_and_labels(
    annotations_df, 
    col, 
    model, 
    num_none_included,
    seed=42,
):
    df = annotations_df.dropna(subset=[f'{model}_vis_embs']).copy()
    df = filter_none(df, num_none_included, col, seed=seed)
        
    embeddings_avg = np.array(df[f'{model}_vis_embs'].values.tolist()).mean(axis=1)
    labels = np.array(df[col].values.tolist())
    df['embedding'] = list(embeddings_avg)
    df['label'] = list(labels)
    df = df.reset_index(drop=True)
    return df

def get_embeddings_and_labels(annotations_df, procedures_df, tasks_df, col, model_name):
    df = annotations_df.copy()
    
    # Vision embeddings
    vis_embs_mean = np.array(df[f'{model_name}_vis_embs'].values.tolist()).mean(axis=1)
    print(f"vis_embs_mean.shape: {vis_embs_mean.shape}")
    print(f"df[f'{model_name}_vis_embs'][0].shape: {df[f'{model_name}_vis_embs'].iloc[0].shape}")
    
    emb_col = ''
    if model_name == 'surgvlp': emb_col = 'SurgVLP'
    elif model_name == 'hecvl': emb_col = 'HecVL'
    elif model_name == 'peskavlp': emb_col = 'PeskaVLP'
    elif model_name == 'pe224' or model_name == 'pe336' or model_name == 'pe448': emb_col = 'MedEmbed_small'
    elif 'videomae' in model_name: emb_col = 'MedEmbed_small'
    elif model_name == 'vjepa2': emb_col = 'MedEmbed_small'
    else:
        raise ValueError(f"Model name {model_name} is not supported.")
    print(f"Embedding column: {emb_col}")
    
    # Procedure embeddings
    procedure_embs = []
    if procedures_df is not None:
        for i in range(len(df)):
            case = df.iloc[i]['case']
            tmp_df = procedures_df[procedures_df['case_id'] == case]
            procedure_embs.append(tmp_df[f"procedure_defn_emb-{emb_col}"].values[0])
    procedure_embs = np.array(procedure_embs)
    
    # Tasks embeddings
    task_embs = []
    if tasks_df is not None:
        df['secs'] = df['cvid'].apply(lambda x: sum([a*b for a,b in zip(map(int, x.split('_')[-1][:-4].split('-')), [3600, 60, 1])]))
        for i in range(len(df)):
            secs = df.iloc[i]['secs']
            case = df.iloc[i]['case']
            tmp_df = tasks_df[tasks_df['case_id'] == case]
            tmp_df = tmp_df[(tmp_df['start_secs'] <= secs) & (tmp_df['end_secs'] > secs)]
            emb_size = len(tasks_df[f"task_defn_emb-{emb_col}"].iloc[0])
            if len(tmp_df) == 0:
                # print(f"No teaching step found for case {case} and timestamp {secs}")
                task_embs.append(np.zeros(emb_size))
            elif len(tmp_df) > 1:
                print(f"Multiple teaching steps found for case {case} and timestamp {secs}")
                task_embs.append(np.zeros(emb_size))
            else:
                task_embs.append(tmp_df[f"task_defn_emb-{emb_col}"].values[0])
    task_embs = np.array(task_embs)
    
    # Labels
    labels = np.array(df[col].values.tolist())
    
    df['vis_embs_avg'] = list(vis_embs_mean)
    df['procedure_embs'] = list(procedure_embs) if procedures_df is not None else None
    df['task_embs'] = list(task_embs) if tasks_df is not None else None
    df['label'] = list(labels)
    df = df.reset_index(drop=True)
    
    if procedures_df is not None and tasks_df is not None:
        embeddings_comb = np.concatenate([vis_embs_mean, procedure_embs, task_embs], axis=1)
    elif procedures_df is not None:
        embeddings_comb = np.concatenate([vis_embs_mean, procedure_embs], axis=1)
    elif tasks_df is not None:
        embeddings_comb = np.concatenate([vis_embs_mean, task_embs], axis=1)
    else:
        embeddings_comb = vis_embs_mean
    df['embedding'] = list(embeddings_comb)
    
    # print emb dims
    print(f"Vision Embedding Dimension: {vis_embs_mean.shape[1]}")
    if procedures_df is not None:
        print(f"Procedure Embedding Dimension: {procedure_embs.shape[1]}")
    if tasks_df is not None:
        print(f"Task Embedding Dimension: {task_embs.shape[1]}")
    print(f"Combined Embedding Dimension: {embeddings_comb.shape[1]}")
    
    return df

import numpy as np
import pandas as pd

def get_embeddings_and_labels_multiple(annotations_df, procedures_df, tasks_df, col, model_name, current_fps, target_fps):
    """
    Generates a DataFrame with embeddings and labels, expanding rows for each individual vision embedding.

    This function takes annotations and optional procedure/task dataframes. For each row 
    in the annotations_df, it duplicates the row for every frame-wise vision embedding 
    found in 'vis_embs_multiple'. The final DataFrame contains a row for each individual 
    frame, with a new 'vis_embs_individual' column and a combined 'embedding' column.
    
    Vision embeddings can be downsampled from a `current_fps` to a `target_fps`.

    Args:
        annotations_df (pd.DataFrame): DataFrame containing annotations and a column with lists of vision embeddings.
        procedures_df (pd.DataFrame): Optional DataFrame with procedure text embeddings.
        tasks_df (pd.DataFrame): Optional DataFrame with task text embeddings.
        col (str): The name of the column in annotations_df to use as the label.
        model_name (str): The name of the model used for embeddings, to determine which columns to use.
        current_fps (int): The original frames per second of the vision embeddings.
        target_fps (int): The target frames per second to sample the embeddings to.

    Returns:
        pd.DataFrame: A new DataFrame where each row corresponds to a single vision embedding,
                      containing the individual vision embedding, its corresponding label, any text
                      embeddings, and a final concatenated embedding vector.
    """
    df = annotations_df.copy()
    
    # Vision embeddings are expected to be a list of embedding arrays for each row.
    vis_embs_lists = df[f'{model_name}_vis_embs'].values.tolist()
    
    # Sample the vision embeddings based on target FPS.
    if current_fps > target_fps:
        stride = int(round(current_fps / target_fps))
        if stride > 1:
            print(f"Sampling embeddings from {current_fps}fps to {target_fps}fps with a stride of {stride}.")
            vis_embs_lists = [embs[::stride] for embs in vis_embs_lists]
    
    # Determine the corresponding text embedding column based on the model name.
    emb_col = ''
    if model_name == 'surgvlp': emb_col = 'SurgVLP'
    elif model_name == 'hecvl': emb_col = 'HecVL'
    elif model_name == 'peskavlp': emb_col = 'PeskaVLP'
    elif 'pe' in model_name or 'videomae' in model_name: emb_col = 'MedEmbed_small'
    else:
        raise ValueError(f"Model name {model_name} is not supported.")
    print(f"Embedding column: {emb_col}")
    
    # Extract procedure embeddings if the dataframe is provided.
    procedure_embs = []
    if procedures_df is not None:
        for i in range(len(df)):
            case = df.iloc[i]['case']
            tmp_df = procedures_df[procedures_df['case_id'] == case]
            procedure_embs.append(tmp_df[f"procedure_defn_emb-{emb_col}"].values[0])
    
    # Extract task embeddings if the dataframe is provided.
    task_embs = []
    if tasks_df is not None:
        if 'secs' not in df.columns:
             df['secs'] = df['cvid'].apply(lambda x: sum([a*b for a,b in zip(map(int, x.split('_')[-1][:-4].split('-')), [3600, 60, 1])]))
        
        for i in range(len(df)):
            secs = df.iloc[i]['secs']
            case = df.iloc[i]['case']
            tmp_df = tasks_df[tasks_df['case_id'] == case]
            tmp_df = tmp_df[(tmp_df['start_secs'] <= secs) & (tmp_df['end_secs'] > secs)]
            emb_size = len(tasks_df[f"task_defn_emb-{emb_col}"].iloc[0])
            if len(tmp_df) == 0:
                task_embs.append(np.zeros(emb_size))
            elif len(tmp_df) > 1:
                print(f"Multiple teaching steps found for case {case} and timestamp {secs}")
                task_embs.append(np.zeros(emb_size))
            else:
                task_embs.append(tmp_df[f"task_defn_emb-{emb_col}"].values[0])
    
    # Extract labels.
    labels = df[col].values.tolist()
    
    # --- Main Logic: Expand the DataFrame ---
    # Create a new list of rows, where each original row is duplicated for each of its vision embeddings.
    expanded_rows = []
    for i in range(len(df)):
        original_row_data = df.iloc[i].to_dict()
        current_label = labels[i]
        current_proc_emb = procedure_embs[i] if procedures_df is not None else None
        current_task_emb = task_embs[i] if tasks_df is not None else None
        
        # Iterate over each individual vision embedding for the current row.
        for vis_emb in vis_embs_lists[i]:
            new_row = original_row_data.copy()
            
            # Add the specific vision embedding for this new row.
            new_row['vis_embs_individual'] = vis_emb
            new_row['label'] = current_label

            # Add procedure and task embeddings.
            if procedures_df is not None:
                new_row['procedure_embs'] = current_proc_emb
            if tasks_df is not None:
                new_row['task_embs'] = current_task_emb

            # Create the final combined embedding.
            embs_to_concat = [vis_emb]
            if procedures_df is not None:
                embs_to_concat.append(current_proc_emb)
            if tasks_df is not None:
                embs_to_concat.append(current_task_emb)
            
            new_row['embedding'] = np.concatenate(embs_to_concat)
            expanded_rows.append(new_row)
            
    final_df = pd.DataFrame(expanded_rows).reset_index(drop=True)

    # Print embedding dimensions for verification.
    print(f"Vision Embedding Dimension: {len(final_df['vis_embs_individual'].iloc[0])}")
    if procedures_df is not None:
        print(f"Procedure Embedding Dimension: {len(final_df['procedure_embs'].iloc[0])}")
    if tasks_df is not None:
        print(f"Task Embedding Dimension: {len(final_df['task_embs'].iloc[0])}")
    print(f"Combined Embedding Dimension: {len(final_df['embedding'].iloc[0])}")
    
    return final_df



def run_via_embs(
    iat_col: str,   # instrument, action, or tissue
    model: str, # peskavlp, surgvlp, hecvl, pe224, pe336, pe448, videomae_urology, videomae_cholect45
    inputs: str, # vision, vision+procedure, vision+task, vision
    output_json: str,
    pred_csv: str = None,
    num_none_included: int = 100,
    vision_embeddings_dir: str = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/outputs/embeddings/vision',
    annotations_path: str = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/data/iat_predictor_splits/full.csv',
    procedures_embs_path: str = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/data/iat/procedures_embs_df.parquet',
    tasks_embs_path: str = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/data/iat/tasks_embs_df.parquet',
    seed: int = 0,
    metric_avg: str = 'weighted',
):
    print(f"IAT Column: {iat_col}")
    print(f"Model: {model}")
    print(f"Inputs: {inputs}")
    print(f"Num none included: {num_none_included}")
    
    annotations_df = pd.read_csv(annotations_path).replace('NONE', None)
    annotations_df = add_vis_embs(model, annotations_df, vision_embeddings_dir, force=True)
    annotations_df = annotations_df.dropna(subset=[f'{model}_vis_embs']).copy()
    annotations_df = filter_none(annotations_df, num_none_included, iat_col, seed=seed)
    print(f"Number of samples after filtering: {len(annotations_df)}")
    
    vis_emb_cols = [col for col in annotations_df.columns if col.endswith('_vis_embs')]
    for col in vis_emb_cols:
        print(f"{col}: {annotations_df.iloc[0][col].shape if isinstance(annotations_df.iloc[0][col], np.ndarray) else 'None'}")
   
    if inputs == 'vision':
        processed_df = get_embeddings_and_labels(annotations_df, None, None, iat_col, model)
    elif inputs == 'vision+procedure':
        procedures_df = pd.read_parquet(procedures_embs_path)
        processed_df = get_embeddings_and_labels(annotations_df, procedures_df, None, iat_col, model)
    elif inputs == 'vision+task':
        tasks_df = pd.read_parquet(tasks_embs_path)
        processed_df = get_embeddings_and_labels(annotations_df, None, tasks_df, iat_col, model)
    elif inputs == 'vision+procedure+task':
        procedures_df = pd.read_parquet(procedures_embs_path)
        tasks_df = pd.read_parquet(tasks_embs_path)
        processed_df = get_embeddings_and_labels(annotations_df, procedures_df, tasks_df, iat_col, model)
    else:
        raise ValueError(f"Inputs type {inputs} is not supported.")
    metrics, pred_df = evaluate_via_embs(processed_df, metric_avg=metric_avg, num_folds=5, hidden_layer_sizes=(64, 32, 16), seed=seed)

    print(f"Mean AUPRC: {metrics['auprc_mean']:.4f}")
    print(f"Mean AUROC: {metrics['auroc_mean']:.4f}")
    print(f"Mean Accuracy: {metrics['accuracy_mean']:.4f}")
    print(f"Mean Precision: {metrics['precision_mean']:.4f}")
    print(f"Mean Recall: {metrics['recall_mean']:.4f}")
    print(f"Mean F1: {metrics['f1_mean']:.4f}")
    print(f"Mean ECE: {metrics['ece_mean']:.4f}")
    with open(output_json, 'w') as f:
        metrics['iat_col'] = iat_col
        metrics['model'] = model
        metrics['inputs'] = inputs
        metrics['num_none_included'] = num_none_included
        
        json.dump(metrics, f, indent=4)
    
    if pred_csv is not None:
        pred_df.to_csv(pred_csv, index=False)

import numpy as np

def standardize_track_shape(track, target_dots=None):
    if target_dots is None:
        target_dots = 400
    num_frames = track.shape[0]
    current_dots = track.shape[1]

    target_shape = (num_frames, target_dots, 2)
    standardized_track = np.zeros(target_shape, dtype=track.dtype)
    
    dots_to_copy = min(current_dots, target_dots)
    
    standardized_track[:, :dots_to_copy, :] = track[:, :dots_to_copy, :]
    return standardized_track

def _calculate_ece(y_true, y_prob, num_bins=5):
    """
    Computes the Expected Calibration Error (ECE).
    This version matches the logic of the user-provided example.

    Args:
        y_true: The true labels for the samples.
        y_prob: The predicted probabilities for each class (shape: [n_samples, n_classes]).
        num_bins: The number of bins to use for the calculation.
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

class HybridDataset(Dataset):
    def __init__(self, embeddings, tracks, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.tracks = torch.tensor(tracks, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.tracks[idx], self.labels[idx]


class LSTMFusionModel(nn.Module):
    def __init__(self, embedding_dim, track_feature_dim, num_classes, lstm_hidden_dim=64, num_lstm_layers=1, dropout=0.2):
        super(LSTMFusionModel, self).__init__()
        
        # LSTM to process the instrument track sequence
        self.lstm = nn.LSTM(
            input_size=track_feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True
        )
        
        # Simple MLP classification head
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
        # embedding shape: (batch, embedding_dim)
        # track shape: (batch, seq_len, track_feature_dim)
        
        # Process track with LSTM
        # We only need the final hidden state, so we ignore the outputs
        _, (hidden_state, _) = self.lstm(track)
        
        # Get the hidden state from the last layer, shape: (batch, lstm_hidden_dim)
        track_summary = hidden_state[-1]
        
        # Concatenate the global embedding with the track summary
        combined_features = torch.cat((embedding, track_summary), dim=1)
        
        # Get final predictions
        logits = self.classification_head(combined_features)
        return logits
    

class CrossTransformerModel(nn.Module):
    def __init__(self, embedding_dim, track_feature_dim, num_heads, num_classes, hidden_dim=128, num_layers=1, dropout=0.2):
        super(CrossTransformerModel, self).__init__()
        self.embedding_proj = nn.Linear(embedding_dim, hidden_dim)
        self.track_proj = nn.Linear(track_feature_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Dynamic sinusoidal positional embeddings (no parameters to slim down here)
        max_len = 200
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(1, max_len, hidden_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_embedding', pe)
        
        self.pos_dropout = nn.Dropout(p=dropout)

        # A single, lighter transformer layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, # Reduced from 4 to 2
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # A much simpler classification head
        self.classification_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, embedding, track):
        emb_proj = self.embedding_proj(embedding).unsqueeze(1)
        track_proj = self.track_proj(track)
        
        cls_tokens = self.cls_token.expand(embedding.shape[0], -1, -1)
        
        x = torch.cat((cls_tokens + emb_proj, track_proj), dim=1)
        
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.pos_dropout(x)
        
        transformer_output = self.transformer_encoder(x)
        cls_output = transformer_output[:, 0, :]
        
        logits = self.classification_head(cls_output)
        return logits

    
class HybridLSTMTransformerModel(nn.Module):
    def __init__(self, embedding_dim, track_feature_dim, num_classes, 
                 lstm_hidden_dim=64, num_lstm_layers=1, 
                 hidden_dim=128, num_heads=2, num_layers=1, dropout=0.2):
        super(HybridLSTMTransformerModel, self).__init__()
        
        # --- Part 1: LSTM for Track Summarization (from LSTMFusionModel) ---
        self.lstm = nn.LSTM(
            input_size=track_feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True
        )
        
        # --- Part 2: Cross-Attention Transformer (adapted from CrossTransformerModel) ---
        self.embedding_proj = nn.Linear(embedding_dim, hidden_dim)
        
        #  projects the LSTM's output, not the raw track features.
        self.track_proj = nn.Linear(lstm_hidden_dim, hidden_dim) 
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Using fixed sinusoidal positional embeddings for the two inputs (CLS and track summary)
        max_len = 10 # More than enough for CLS + track summary
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(1, max_len, hidden_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_embedding', pe)
        
        self.pos_dropout = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classification_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, embedding, track):
        # 1. Process the entire track sequence with the LSTM to get a summary vector
        # We only need the final hidden state.
        _, (hidden_state, _) = self.lstm(track)
        
        # Get the hidden state from the last LSTM layer, shape: (batch, lstm_hidden_dim)
        track_summary = hidden_state[-1]
        
        # 2. Prepare inputs for the Transformer
        emb_proj = self.embedding_proj(embedding).unsqueeze(1) # Shape: (batch, 1, hidden_dim)
        
        # Project the track summary and add a sequence dimension
        track_proj = self.track_proj(track_summary).unsqueeze(1) # Shape: (batch, 1, hidden_dim)
        
        # Prepare the classification token
        cls_tokens = self.cls_token.expand(embedding.shape[0], -1, -1) # Shape: (batch, 1, hidden_dim)
        
        # 3. Combine for the transformer's input sequence (length = 2)
        # The sequence is now [CLS_token+embedding, track_summary]
        x = torch.cat((cls_tokens + emb_proj, track_proj), dim=1)
        
        # Add positional embeddings for the two items in the sequence
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.pos_dropout(x)
        
        # 4. Pass through the transformer encoder
        transformer_output = self.transformer_encoder(x)
        
        # 5. Get the output corresponding to the CLS token for classification
        cls_output = transformer_output[:, 0, :]
        
        # 6. Final classification
        logits = self.classification_head(cls_output)
        return logits


import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, embedding_dim, track_feature_dim, num_classes, track_summary_dim=64, embedding_summary_dim=64, hidden_dims=[128, 64], dropout=0.2):
        """
        Initializes the FFNFusionModel.

        Args:
            embedding_dim (int): The dimensionality of the global input embedding.
            track_feature_dim (int): The dimensionality of features in the track sequence.
            num_classes (int): The number of output classes for classification.
            track_summary_dim (int): The output dimension for the track processing MLP.
            embedding_summary_dim (int): The output dimension for the embedding processing MLP.
            hidden_dims (list of int, optional): A list specifying the size of each
                                                 hidden layer in the final classification head.
                                                 Defaults to [128, 64].
            dropout (float, optional): The dropout rate to apply after hidden layers.
                                       Defaults to 0.2.
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

        # The input dimension for the final FFN is the sum of the processed feature dimensions.
        input_dim = track_summary_dim + embedding_summary_dim
        
        layers = []
        # Dynamically build the final MLP layers based on the hidden_dims list
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            # The input for the next layer is the output of this one
            input_dim = h_dim
        
        # Add the final output layer
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
        # embedding shape: (batch_size, embedding_dim)
        # track shape: (batch_size, seq_len, track_feature_dim)

        # 1. Create track summary by averaging over the sequence length dimension.
        # track_mean shape: (batch_size, track_feature_dim)
        track_mean = torch.mean(track, dim=1)
        # track_summary shape: (batch_size, track_summary_dim)
        track_summary = self.track_processor(track_mean)

        # 2. Process the global embedding.
        # processed_embedding shape: (batch_size, embedding_summary_dim)
        processed_embedding = self.embedding_processor(embedding)
        
        # 3. Concatenate the processed embedding with the track summary.
        # combined_features shape: (batch_size, embedding_summary_dim + track_summary_dim)
        combined_features = torch.cat((processed_embedding, track_summary), dim=1)
        
        # 4. Get final predictions from the MLP head.
        # logits shape: (batch_size, num_classes)
        logits = self.classification_head(combined_features)
        
        return logits




def evaluate_via_hybrid_original(
    processed_df,
    metric_avg='weighted',
    sampler=None,
    num_folds=5,
    seed=42,
    **kwargs
):
    """
    Evaluates the hybrid model using stratified k-fold cross-validation,
    adds prediction confidence to the output dataframe, and computes ECE.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    embeddings = np.array(processed_df['embedding'].values.tolist())
    tracks = np.array(processed_df['tracks'].values.tolist())
    labels = processed_df['label'].values
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    # try:
    #     none_label_encoded = label_encoder.transform(['None'])[0]
    #     print(f"The label 'None' is encoded as: {none_label_encoded}. It will be ignored during loss calculation.")
    # except ValueError:
    #     none_label_encoded = -1 
    #     print("The label 'None' was not found. No labels will be masked during training.")
    
    
    pred_df = processed_df.copy()
    pred_df['confidence'] = 0.0  # Initialize confidence column
    tracks_reshaped = tracks.reshape(tracks.shape[0], tracks.shape[1], -1)
    
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    
    auprc_scores, auroc_scores, accuracy_scores = [], [], []
    precision_scores, recall_scores, f1_scores = [], [], []
    ece_scores = []
    confusion_matrices = []

    epochs=20
    batch_size=8
    learning_rate=0.001
    
    fold_num = 1
    for train_index, test_index in skf.split(tracks_reshaped, encoded_labels):
        print(f"Fold {fold_num}/{num_folds}")
        fold_num += 1

        X_train_embs, X_test_embs = embeddings[train_index], embeddings[test_index]
        X_train_tracks, X_test_tracks = tracks_reshaped[train_index], tracks_reshaped[test_index]
        y_train, y_test = encoded_labels[train_index], encoded_labels[test_index]

        train_dataset = HybridDataset(X_train_embs, X_train_tracks, y_train)
        test_dataset = HybridDataset(X_test_embs, X_test_tracks, y_test)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        
        embedding_dim = X_train_embs.shape[1]
        track_feature_dim = X_train_tracks.shape[2]
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = LSTMFusionModel(
            embedding_dim=embedding_dim,
            track_feature_dim=track_feature_dim,
            num_classes=num_classes,
            lstm_hidden_dim=32,
            num_lstm_layers=10,
            dropout=0.2
        ).to(device)
        
        # determine class weights by sqrt of counts
        criterion = nn.CrossEntropyLoss()
        criterion_weighted = nn.CrossEntropyLoss(
            weight=1.0 / torch.sqrt(torch.tensor(np.bincount(y_train), dtype=torch.float32).to(device))
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            torch.manual_seed(seed + epoch)
            torch.cuda.manual_seed_all(seed + epoch)
            model.train()
            for batch_embs, batch_tracks, batch_labels in train_loader:
                batch_embs, batch_tracks, batch_labels = batch_embs.to(device), batch_tracks.to(device), batch_labels.to(device)
                
                # mask = batch_labels != none_label_encoded
                # if mask.sum() > 0:
                #     outputs = model(batch_embs, batch_tracks)
                #     loss = criterion(outputs[mask], batch_labels[mask])
                #     optimizer.zero_grad()
                #     loss.backward()
                #     optimizer.step()
                
                outputs = model(batch_embs, batch_tracks)
                loss = criterion(outputs, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        if kwargs.get('do_decoupling', True):
            print("Applying the training trick: freezing feature encoder and training clf head only.")
            # freeze feature encoder and train clf head
            decoupled_epochs = 3
            for name, param in model.lstm.named_parameters():
                param.requires_grad = False
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1 - (step / (decoupled_epochs * len(train_loader))))
            global_step = 0
            for epoch in range(decoupled_epochs):
                torch.manual_seed(seed + epoch)
                torch.cuda.manual_seed_all(seed + epoch)
                model.train()
                batch_idx = 0
                for batch_embs, batch_tracks, batch_labels in train_loader:
                    batch_embs, batch_tracks, batch_labels = batch_embs.to(device), batch_tracks.to(device), batch_labels.to(device)
                    
                    outputs = model(batch_embs, batch_tracks)
                    loss = criterion(outputs, batch_labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    global_step += 1
                    batch_idx += 1
                    
                    # mask = batch_labels != none_label_encoded
                    # if mask.sum() > 0:
                    #     outputs = model(batch_embs, batch_tracks)
                    #     loss = criterion(outputs[mask], batch_labels[mask])
                    #     optimizer.zero_grad()
                    #     loss.backward()
                    #     optimizer.step()
                    #     scheduler.step()
                    #     global_step += 1
                    # batch_idx += 1

        model.eval()
        all_preds, all_probs, all_labels, all_confidences = [], [], [], [] # <-- Added all_confidences
        with torch.no_grad():
            for batch_embs, batch_tracks, batch_labels in test_loader:
                outputs = model(batch_embs.to(device), batch_tracks.to(device))
                probs = torch.softmax(outputs, dim=1)
                
                # Get both confidence and predicted class
                batch_confidences, predicted = torch.max(probs, 1)

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_confidences.extend(batch_confidences.cpu().numpy())

        y_pred_prob, y_pred, y_test_fold = np.array(all_probs), np.array(all_preds), np.array(all_labels)
        y_confidences = np.array(all_confidences)

        # Add predictions and confidences to the dataframe
        pred_df.loc[test_index, 'pred'] = label_encoder.inverse_transform(y_pred)
        pred_df.loc[test_index, 'confidence'] = y_confidences
        for i, idx in enumerate(test_index):
            pred_df.loc[idx, 'pred_probs'] = str({label_encoder.classes_[j]: float(y_pred_prob[i][j]) for j in range(num_classes)})

        # Calculate metrics for the fold
        auprc_scores.append(average_precision_score(y_test_fold, y_pred_prob, average=metric_avg))
        auroc_scores.append(roc_auc_score(y_test_fold, y_pred_prob, multi_class='ovo', average=metric_avg))
        accuracy_scores.append(accuracy_score(y_test_fold, y_pred))
        precision_scores.append(precision_score(y_test_fold, y_pred, average=metric_avg, zero_division=0))
        recall_scores.append(recall_score(y_test_fold, y_pred, average=metric_avg, zero_division=0))
        f1_scores.append(f1_score(y_test_fold, y_pred, average=metric_avg, zero_division=0))
        ece_scores.append(_calculate_ece(y_test_fold, y_pred_prob))
        confusion_matrices.append(compute_confusion_matrix(y_test_fold, y_pred, labels=np.arange(num_classes)))

    pred_df = pred_df.drop(columns=[col for col in pred_df.columns if 'emb' in col] + ['tracks'])

    mean_auprc, mean_auroc = float(np.mean(auprc_scores)), float(np.mean(auroc_scores))
    mean_accuracy, mean_precision = float(np.mean(accuracy_scores)), float(np.mean(precision_scores))
    mean_recall, mean_f1 = float(np.mean(recall_scores)), float(np.mean(f1_scores))
    mean_ece = float(np.mean(ece_scores)) # <-- Calculate mean ECE
        
    confusion_matrix = np.sum(confusion_matrices, axis=0)
    confusion_matrix = [list(map(int, row)) for row in confusion_matrix]

    metrics = {
        'auprc_scores': auprc_scores, 'auroc_scores': auroc_scores, 'accuracy_scores': accuracy_scores,
        'precision_scores': precision_scores, 'recall_scores': recall_scores, 'f1_scores': f1_scores,
        'ece_scores': ece_scores,
        'auprc_mean': mean_auprc, 'auroc_mean': mean_auroc, 'accuracy_mean': mean_accuracy,
        'precision_mean': mean_precision, 'recall_mean': mean_recall, 'f1_mean': mean_f1,
        'ece_mean': mean_ece,
        'confusion_matrix': confusion_matrix, 'confusion_matrix_labels': label_encoder.classes_.tolist(),
    }
    return metrics, pred_df


def evaluate_via_hybrid(
    processed_df,
    metric_avg='weighted',
    sampler=None,
    num_folds=5,
    seed=42,
    **kwargs
):
    """
    Evaluates the hybrid model using stratified k-fold cross-validation,
    adds prediction confidence to the output dataframe, and computes ECE.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    embeddings = np.array(processed_df['embedding'].values.tolist())
    tracks = np.array(processed_df['tracks'].values.tolist())
    labels = processed_df['label'].values
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    pred_df = processed_df.copy()
    pred_df['confidence'] = 0.0  # Initialize confidence column
    tracks_reshaped = tracks.reshape(tracks.shape[0], tracks.shape[1], -1)
    
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    
    auprc_scores, auroc_scores, accuracy_scores = [], [], []
    precision_scores, recall_scores, f1_scores = [], [], []
    ece_scores = []
    confusion_matrices = []

    epochs=20
    batch_size=8
    learning_rate=0.001
    
    fold_num = 1
    for train_index, test_index in skf.split(tracks_reshaped, encoded_labels):
        print(f"Fold {fold_num}/{num_folds}")
        fold_num += 1

        X_train_embs, X_test_embs = embeddings[train_index], embeddings[test_index]
        X_train_tracks, X_test_tracks = tracks_reshaped[train_index], tracks_reshaped[test_index]
        y_train, y_test = encoded_labels[train_index], encoded_labels[test_index]

        train_dataset = HybridDataset(X_train_embs, X_train_tracks, y_train)
        test_dataset = HybridDataset(X_test_embs, X_test_tracks, y_test)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        
        embedding_dim = X_train_embs.shape[1]
        track_feature_dim = X_train_tracks.shape[2]
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = LSTMFusionModel(
            embedding_dim=embedding_dim,
            track_feature_dim=track_feature_dim,
            num_classes=num_classes,
            lstm_hidden_dim=32,
            num_lstm_layers=10,
            dropout=0.2
        ).to(device)
        # model = CrossTransformerModel(
        #     embedding_dim=embedding_dim,
        #     track_feature_dim=track_feature_dim,
        #     num_classes=num_classes,
        #     num_heads=2,
        #     num_layers=4,
        # ).to(device)
        # model = FFNFusionModel(
        #     embedding_dim=embedding_dim,
        #     track_feature_dim=track_feature_dim,
        #     num_classes=num_classes,
        #     track_summary_dim=64,
        #     embedding_summary_dim=64,
        #     hidden_dims=(64, 16),
        #     dropout=0.2,
        # ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            torch.manual_seed(seed + epoch)
            torch.cuda.manual_seed_all(seed + epoch)
            model.train()
            for batch_embs, batch_tracks, batch_labels in train_loader:
                batch_embs, batch_tracks, batch_labels = batch_embs.to(device), batch_tracks.to(device), batch_labels.to(device)
                
                outputs = model(batch_embs, batch_tracks)
                loss = criterion(outputs, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # if kwargs.get('do_decoupling', True):
        #     print("Applying the training trick: freezing feature encoder and training clf head only.")
        #     decoupled_epochs = 3
        #     for name, param in model.lstm.named_parameters():
        #         param.requires_grad = False
        #     optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
        #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1 - (step / (decoupled_epochs * len(train_loader))))
        #     for epoch in range(decoupled_epochs):
        #         torch.manual_seed(seed + epoch)
        #         torch.cuda.manual_seed_all(seed + epoch)
        #         model.train()
        #         for batch_embs, batch_tracks, batch_labels in train_loader:
        #             batch_embs, batch_tracks, batch_labels = batch_embs.to(device), batch_tracks.to(device), batch_labels.to(device)
                    
        #             outputs = model(batch_embs, batch_tracks)
        #             loss = criterion(outputs, batch_labels)
        #             optimizer.zero_grad()
        #             loss.backward()
        #             optimizer.step()
        #             scheduler.step()
        
        model.eval()
        
        all_fold_embeddings = []
        all_fold_track_summaries = []
        
        full_fold_dataset = HybridDataset(embeddings, tracks_reshaped, encoded_labels)
        full_fold_loader = DataLoader(dataset=full_fold_dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch_embs, batch_tracks, _ in full_fold_loader:
                batch_tracks_device = batch_tracks.to(device)
                if isinstance(model, LSTMFusionModel):
                    _, (hidden_state, _) = model.lstm(batch_tracks_device)
                    track_summary = hidden_state[-1]
                elif isinstance(model, CrossTransformerModel):
                    hidden_state = model.track_proj(batch_tracks_device)
                    track_summary = hidden_state[:, -1, :]
                elif isinstance(model, FFNFusionModel):
                    track_mean = torch.mean(batch_tracks_device, dim=1)
                    track_summary = model.track_processor(track_mean)
                
                all_fold_track_summaries.extend(track_summary.cpu().numpy())
                all_fold_embeddings.extend(batch_embs.cpu().numpy())
        
        all_fold_embeddings = np.array(all_fold_embeddings)
        all_fold_track_summaries = np.array(all_fold_track_summaries)
        print(f"Motion embeddings shape: {all_fold_track_summaries.shape}")
        
        combined_features = np.concatenate((all_fold_embeddings, all_fold_track_summaries), axis=1)

        X_train_mlp, X_test_mlp = combined_features[train_index], combined_features[test_index]
        
        mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='relu', max_iter=10000, random_state=seed)
        mlp_model.fit(X_train_mlp, y_train)

        y_pred = mlp_model.predict(X_test_mlp)
        y_pred_prob = mlp_model.predict_proba(X_test_mlp)
        y_confidences = np.max(y_pred_prob, axis=1)
        y_test_fold = y_test

        pred_df.loc[test_index, 'pred'] = label_encoder.inverse_transform(y_pred)
        pred_df.loc[test_index, 'confidence'] = y_confidences
        for i, idx in enumerate(test_index):
            pred_df.loc[idx, 'pred_probs'] = str({label_encoder.classes_[j]: float(y_pred_prob[i][j]) for j in range(num_classes)})

        y_test_binarized = label_binarize(y_test_fold, classes=np.arange(num_classes))

        auprc_scores.append(average_precision_score(y_test_binarized, y_pred_prob, average=metric_avg))
        auroc_scores.append(roc_auc_score(y_test_fold, y_pred_prob, multi_class='ovo', average='weighted'))
        accuracy_scores.append(accuracy_score(y_test_fold, y_pred))
        precision_scores.append(precision_score(y_test_fold, y_pred, average=metric_avg, zero_division=0))
        recall_scores.append(recall_score(y_test_fold, y_pred, average=metric_avg, zero_division=0))
        f1_scores.append(f1_score(y_test_fold, y_pred, average=metric_avg, zero_division=0))
        ece_scores.append(_calculate_ece(y_test_fold, y_pred_prob))
        confusion_matrices.append(compute_confusion_matrix(y_test_fold, y_pred, labels=np.arange(num_classes)))

    pred_df = pred_df.drop(columns=[col for col in pred_df.columns if 'emb' in col] + ['tracks'])

    mean_auprc, mean_auroc = float(np.mean(auprc_scores)), float(np.mean(auroc_scores))
    mean_accuracy, mean_precision = float(np.mean(accuracy_scores)), float(np.mean(precision_scores))
    mean_recall, mean_f1 = float(np.mean(recall_scores)), float(np.mean(f1_scores))
    mean_ece = float(np.mean(ece_scores))
        
    confusion_matrix = np.sum(confusion_matrices, axis=0)
    confusion_matrix = [list(map(int, row)) for row in confusion_matrix]

    metrics = {
        'auprc_scores': auprc_scores, 'auroc_scores': auroc_scores, 'accuracy_scores': accuracy_scores,
        'precision_scores': precision_scores, 'recall_scores': recall_scores, 'f1_scores': f1_scores,
        'ece_scores': ece_scores,
        'auprc_mean': mean_auprc, 'auroc_mean': mean_auroc, 'accuracy_mean': mean_accuracy,
        'precision_mean': mean_precision, 'recall_mean': mean_recall, 'f1_mean': mean_f1,
        'ece_mean': mean_ece,
        'confusion_matrix': confusion_matrix, 'confusion_matrix_labels': label_encoder.classes_.tolist(),
    }
    return metrics, pred_df


def evaluate_via_hybrid_multiple(
    processed_df,
    metric_avg='weighted',
    sampler=None,
    num_folds=5,
    seed=42,
    **kwargs
):
    """
    Evaluates a hybrid model using instance-level stratified k-fold cross-validation.

    Training is performed on individual frames. For evaluation, predictions are made
    on all frames of a test instance, and the final label is determined by a
    majority vote. The final prediction dataframe has one row per instance.

    Args:
        processed_df (pd.DataFrame): DataFrame where each row corresponds to a single
                                     frame/embedding. Must contain a 'cvid' column
                                     to identify unique feedback instances.
        metric_avg (str): The averaging method for multi-class metrics.
        num_folds (int): The number of folds for cross-validation.
        seed (int): The random seed for reproducibility.
        **kwargs: Additional arguments.

    Returns:
        tuple: A tuple containing:
            - metrics (dict): A dictionary of evaluation metrics.
            - pred_df (pd.DataFrame): A DataFrame with instance-level predictions.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data Preparation ---
    # Extract frame-level data
    embeddings = np.array(processed_df['embedding'].values.tolist())
    tracks = np.array(processed_df['tracks'].values.tolist())
    tracks_reshaped = tracks.reshape(tracks.shape[0], tracks.shape[1], -1)

    # Encode labels
    label_encoder = LabelEncoder()
    processed_df['encoded_label'] = label_encoder.fit_transform(processed_df['label'])
    encoded_labels = processed_df['encoded_label'].values
    num_classes = len(label_encoder.classes_)

    # --- Instance-level Stratified K-Fold ---
    # We must split by instance ('cvid') to prevent data leakage
    unique_instances = processed_df.drop_duplicates(subset=['cvid']).copy()
    instance_cvids = unique_instances['cvid'].values
    instance_labels = unique_instances['encoded_label'].values

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    # --- Metric Storage ---
    all_metrics = collections.defaultdict(list)
    
    # This df will store instance-level predictions from all folds
    final_pred_df = unique_instances.copy()
    final_pred_df['pred'] = None
    final_pred_df['confidence'] = 0.0

    epochs = 20
    batch_size = 8
    learning_rate = 0.001

    fold_num = 1
    for train_instance_indices, test_instance_indices in skf.split(instance_cvids, instance_labels):
        print(f"--- Fold {fold_num}/{num_folds} ---")
        
        # Get the 'cvid's for train and test sets for this fold
        train_cvids = instance_cvids[train_instance_indices]
        test_cvids = instance_cvids[test_instance_indices]

        # Get the row indices from the main processed_df corresponding to these cvids
        train_index = processed_df[processed_df['cvid'].isin(train_cvids)].index
        test_index = processed_df[processed_df['cvid'].isin(test_cvids)].index

        # --- Training (Frame-level) ---
        X_train_embs, X_train_tracks = embeddings[train_index], tracks_reshaped[train_index]
        y_train = encoded_labels[train_index]
        
        train_dataset = HybridDataset(X_train_embs, X_train_tracks, y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        embedding_dim = X_train_embs.shape[1]
        track_feature_dim = X_train_tracks.shape[2]

        # Initialize model for the fold
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = LSTMFusionModel(
            embedding_dim=embedding_dim,
            track_feature_dim=track_feature_dim,
            num_classes=num_classes,
            lstm_hidden_dim=32,
            num_lstm_layers=10,
            dropout=0.2
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            for batch_embs, batch_tracks, batch_labels in train_loader:
                batch_embs, batch_tracks, batch_labels = batch_embs.to(device), batch_tracks.to(device), batch_labels.to(device)
                outputs = model(batch_embs, batch_tracks)
                loss = criterion(outputs, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # --- Evaluation (Frame-level Prediction, then Instance-level Voting) ---
        model.eval()
        X_test_embs, X_test_tracks = embeddings[test_index], tracks_reshaped[test_index]
        y_test_frames = encoded_labels[test_index]
        test_cvids_frames = processed_df.loc[test_index, 'cvid'].values

        test_dataset = HybridDataset(X_test_embs, X_test_tracks, y_test_frames)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        
        fold_preds_probs = []
        with torch.no_grad():
            for batch_embs, batch_tracks, _ in test_loader:
                batch_embs, batch_tracks = batch_embs.to(device), batch_tracks.to(device)
                outputs = model(batch_embs, batch_tracks)
                probs = torch.softmax(outputs, dim=1)
                fold_preds_probs.append(probs.cpu().numpy())
        
        y_pred_prob_frames = np.concatenate(fold_preds_probs, axis=0)
        y_pred_frames = np.argmax(y_pred_prob_frames, axis=1)
        
        # --- Aggregation by Majority Vote ---
        frame_results_df = pd.DataFrame({
            'cvid': test_cvids_frames,
            'true_label': y_test_frames,
            'pred_label': y_pred_frames
        })
        # Add probabilities for each class to the frame-level dataframe
        for i, class_name in enumerate(label_encoder.classes_):
            frame_results_df[f'prob_{class_name}'] = y_pred_prob_frames[:, i]
            
        # Define aggregation logic
        agg_funcs = {
            'pred_label': ('pred_label', lambda x: mode(x, keepdims=False)[0]),
            'true_label': ('true_label', 'first')
        }
        # Add mean aggregation for each probability column
        for class_name in label_encoder.classes_:
            agg_funcs[f'prob_{class_name}'] = (f'prob_{class_name}', 'mean')

        instance_results_df = frame_results_df.groupby('cvid').agg(**agg_funcs).reset_index()
        
        y_test_instance = instance_results_df['true_label'].values
        y_pred_instance = instance_results_df['pred_label'].values
        prob_cols = [f'prob_{c}' for c in label_encoder.classes_]
        y_pred_prob_instance = instance_results_df[prob_cols].values

        # --- Store Instance-level Predictions for final_pred_df ---
        for _, row in instance_results_df.iterrows():
            cvid = row['cvid']
            # Find the index in the final (unique) prediction dataframe
            target_idx = final_pred_df[final_pred_df['cvid'] == cvid].index[0]
            
            final_pred_df.loc[target_idx, 'pred'] = label_encoder.inverse_transform([row['pred_label']])[0]
            final_pred_df.loc[target_idx, 'confidence'] = y_pred_prob_instance[instance_results_df['cvid'] == cvid].max()
            
            # Store full probability distribution
            prob_dict = {label_encoder.classes_[j]: float(y_pred_prob_instance[instance_results_df['cvid'] == cvid][0, j]) for j in range(num_classes)}
            final_pred_df.loc[target_idx, 'pred_probs'] = str(prob_dict)

        # --- Calculate and Store Metrics for the Fold ---
        y_test_binarized = label_binarize(y_test_instance, classes=np.arange(num_classes))
        
        all_metrics['auprc_scores'].append(average_precision_score(y_test_binarized, y_pred_prob_instance, average=metric_avg))
        all_metrics['auroc_scores'].append(roc_auc_score(y_test_instance, y_pred_prob_instance, multi_class='ovo', average='weighted'))
        all_metrics['accuracy_scores'].append(accuracy_score(y_test_instance, y_pred_instance))
        all_metrics['precision_scores'].append(precision_score(y_test_instance, y_pred_instance, average=metric_avg, zero_division=0))
        all_metrics['recall_scores'].append(recall_score(y_test_instance, y_pred_instance, average=metric_avg, zero_division=0))
        all_metrics['f1_scores'].append(f1_score(y_test_instance, y_pred_instance, average=metric_avg, zero_division=0))
        all_metrics['ece_scores'].append(_calculate_ece(y_test_instance, y_pred_prob_instance))
        
        fold_num += 1

    # --- Finalize Metrics ---
    metrics = {f"{key.split('_')[0]}_mean": float(np.mean(val)) for key, val in all_metrics.items()}
    metrics.update(all_metrics) # Also include the list of scores per fold
    
    # Clean up final prediction dataframe
    cols_to_drop = [col for col in final_pred_df.columns if 'emb' in col or col == 'encoded_label'] + ['tracks']
    final_pred_df = final_pred_df.drop(columns=cols_to_drop, errors='ignore')

    return metrics, final_pred_df

# evaluate_via_hybrid = evaluate_via_hybrid_original

def run_via_hybrid(
    iat_col: str,
    model: str, 
    inputs: str,
    output_json: str,
    pred_csv: str = None,
    num_none_included: int = 100,
    vision_embeddings_dir: str = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/outputs/embeddings/vision',
    annotations_path: str = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/data/iat_predictor_splits/full.csv',
    procedures_embs_path: str = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/data/iat/procedures_embs_df.parquet',
    tasks_embs_path: str = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/data/iat/tasks_embs_df.parquet',
    instrument_tracks_path: str = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/outputs/instrument_tracks/instrument_tracks.h5',
    seed: int = 0,
    num_tracks: int = 15,
    uncertainty_calibration: str = None,  # None or 'platt'
    metric_avg: str = 'weighted',
    multiple_instance_training: bool = False,
):
    print(f"IAT Column: {iat_col}")
    print(f"Model: {model}")
    print(f"Inputs: {inputs} + tracks")
    print(f"Num none included: {num_none_included}")
    print(f"Uncertainty calibration: {uncertainty_calibration}")
    print(f"Num tracks: {num_tracks}")
    
    annotations_df = pd.read_csv(annotations_path).replace('NONE', None)
    annotations_df = add_vis_embs(model, annotations_df, vision_embeddings_dir, force=True)
    annotations_df = annotations_df.dropna(subset=[f'{model}_vis_embs']).copy()
    annotations_df = filter_none(annotations_df, num_none_included, iat_col, seed=seed)

    if inputs == 'vision':
        if not multiple_instance_training:
            processed_df = get_embeddings_and_labels(annotations_df, None, None, iat_col, model)
        else:
            processed_df = get_embeddings_and_labels_multiple(annotations_df, None, None, iat_col, model, current_fps=5, target_fps=1)
    elif inputs == 'vision+procedure':
        procedures_df = pd.read_parquet(procedures_embs_path)
        if not multiple_instance_training:
            processed_df = get_embeddings_and_labels(annotations_df, procedures_df, None, iat_col, model)
        else:
            processed_df = get_embeddings_and_labels_multiple(annotations_df, procedures_df, None, iat_col, model, current_fps=5, target_fps=1)
    elif inputs == 'vision+task':
        tasks_df = pd.read_parquet(tasks_embs_path)
        if not multiple_instance_training:
            processed_df = get_embeddings_and_labels(annotations_df, None, tasks_df, iat_col, model)
        else:
            processed_df = get_embeddings_and_labels_multiple(annotations_df, None, tasks_df, iat_col, model, current_fps=5, target_fps=1)
    elif inputs == 'vision+procedure+task':
        procedures_df = pd.read_parquet(procedures_embs_path)
        tasks_df = pd.read_parquet(tasks_embs_path)
        if not multiple_instance_training:
            processed_df = get_embeddings_and_labels(annotations_df, procedures_df, tasks_df, iat_col, model)
        else:
            processed_df = get_embeddings_and_labels_multiple(annotations_df, procedures_df, tasks_df, iat_col, model, current_fps=5, target_fps=1)
    else:
        raise ValueError(f"Inputs type {inputs} is not supported.")
    
    h5 = h5py.File(instrument_tracks_path, 'r')
    processed_df['tracks'] = processed_df['cvid'].apply(lambda x: standardize_track_shape(h5[x][:][0], target_dots=num_tracks) if x in h5 else None)
    h5.close()
    print(processed_df['tracks'].iloc[0].shape) 
    
    processed_df.dropna(subset=['tracks', 'embedding'], inplace=True)
    processed_df = processed_df.reset_index(drop=True)
    print(f"Number of samples after loading embeddings and tracks: {len(processed_df)}")

    eval_func = None
    if not multiple_instance_training:
        eval_func = evaluate_via_hybrid
    else:
        eval_func = evaluate_via_hybrid_multiple
        
    metrics, pred_df = eval_func(
        processed_df, 
        metric_avg=metric_avg, 
        seed=seed, 
        uncertainty_calibration=uncertainty_calibration,
        abstention_mechanism='margin_of_confidence' if uncertainty_calibration is not None else None,
        abstention_threshold=0.15,
        output_calibration_path=output_json.replace('.json', '_calibration.png') if uncertainty_calibration is not None else None,
    )
    

    print(f"Mean AUROC: {metrics['auroc_mean']:.4f}")
    print(f"Mean ECE: {metrics.get('ece_mean', 'N/A')}")
    
    with open(output_json, 'w') as f:
        metrics['iat_col'] = iat_col
        metrics['model'] = model
        metrics['inputs'] = f"{inputs}+tracks"
        metrics['num_none_included'] = num_none_included
        json.dump(metrics, f, indent=4)
        
    if pred_csv is not None:
        pred_df.to_csv(pred_csv, index=False)

    
def main_embs(
    output_format: str = '{iat_col}-{model}-{inputs}-none={num_none_included}',
    num_none_included: int = 100,
    vision_embeddings_dir: str = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/outputs/embeddings/vision',
    annotations_path: str = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/data/iat_predictor_splits/full.csv',
    procedures_embs_path: str = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/data/iat/procedures_embs_df.parquet',
    tasks_embs_path: str = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/data/iat/tasks_embs_df.parquet',
    seed: int = 0,
    metric_avg: str = 'weighted',
):
    for model in [
        # 'peskavlp', 
        # 'surgvlp', 
        # 'hecvl',
        'vjepa2',
    ]:
        for inputs in [
            'vision',
            'vision+procedure',
            'vision+procedure+task',
        ]:
            pred_csvs = {}
            for iat_col in ['instrument', 'action', 'tissue']:
                output_json = os.path.join(
                    '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/outputs/iat_metrics',
                    output_format.format(iat_col=iat_col, model=model, inputs=inputs, num_none_included=num_none_included) + '.json'    
                )
                pred_csv = os.path.join(
                    '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/outputs/iat_predictions',
                    output_format.format(iat_col=iat_col, model=model, inputs=inputs, num_none_included=num_none_included) + '.csv'    
                )
                pred_csvs[iat_col] = pred_csv
                os.makedirs(os.path.dirname(output_json), exist_ok=True)
                os.makedirs(os.path.dirname(pred_csv), exist_ok=True)
                run_via_embs(
                    iat_col=iat_col,
                    model=model,
                    inputs=inputs,
                    output_json=output_json,
                    pred_csv=pred_csv,
                    num_none_included=num_none_included,
                    vision_embeddings_dir=vision_embeddings_dir,
                    annotations_path=annotations_path,
                    procedures_embs_path=procedures_embs_path,
                    tasks_embs_path=tasks_embs_path,
                    seed=seed,
                    metric_avg=metric_avg,
                )
                print('-'*50)
                
            # combine pred_csvs into one
            combined_df = None
            for iat_col, pred_csv in pred_csvs.items():
                df = pd.read_csv(pred_csv)
                df = df[['cvid', 'dialogue', 'procedure', 'procedure_defn', 'task', 'task_defn', 'instrument', 'action', 'tissue', 'pred', 'confidence', 'pred_probs']].rename(
                    columns={
                        'pred': f'{iat_col}_pred',
                        'confidence': f'{iat_col}_conf',
                        'pred_probs': f'{iat_col}_pred_probs',
                    }
                )
                if combined_df is None:
                    combined_df = df
                else:
                    combined_df = combined_df.merge(df, on=['cvid', 'procedure', 'procedure_defn', 'task', 'task_defn', 'dialogue', 'instrument', 'action', 'tissue', ], how='outer')
            combined_pred_csv = os.path.join(
                '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/outputs/iat_predictions',
                output_format.format(iat_col='dummy', model=model, inputs=inputs, num_none_included=num_none_included).replace('dummy-', '') + '.csv'
            )
            combined_df.to_csv(combined_pred_csv, index=False)

def main_hybrid(
    output_format: str = '{iat_col}-{model}-{inputs}+tracks-none={num_none_included}',
    num_none_included: int = 100,
    vision_embeddings_dir: str = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/outputs/embeddings/vision',
    annotations_path: str = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/data/iat_predictor_splits/full.csv',
    procedures_embs_path: str = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/data/iat/procedures_embs_df.parquet',
    tasks_embs_path: str = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/data/iat/tasks_embs_df.parquet',
    instrument_tracks_path: str = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/outputs/instrument_tracks/instrument_tracks-num_tracks=15.h5',
    seed: int = 0,
    num_tracks: int = 15,
    uncertainty_calibration: str = None,  # None or 'platt'
    metric_avg: str = 'weighted',
    multiple_instance_training: bool = False,
):
    for model in [
        # 'peskavlp', 
        # 'surgvlp', 
        # 'hecvl',
        # 'vjepa2',
        'videomae_base',
    ]:
        for inputs in [
            # 'vision',
            # 'vision+procedure',
            'vision+procedure+task',
        ]:
            pred_csvs = {}
            for iat_col in ['instrument', 'action', 'tissue']:
                output_json = os.path.join(
                    '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/outputs/iat_metrics',
                    output_format.format(iat_col=iat_col, model=model, inputs=inputs, num_none_included=num_none_included) + '.json'
                )
                pred_csv = os.path.join(
                    '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/outputs/iat_predictions',
                    output_format.format(iat_col=iat_col, model=model, inputs=inputs, num_none_included=num_none_included) + '.csv'
                )
                
                if uncertainty_calibration is not None:
                    output_json = output_json.replace('.json', f'-uncertainty_calibration={uncertainty_calibration}.json')
                    pred_csv = pred_csv.replace('.csv', f'-uncertainty_calibration={uncertainty_calibration}.csv')
                
                pred_csvs[iat_col] = pred_csv
                os.makedirs(os.path.dirname(output_json), exist_ok=True)
                os.makedirs(os.path.dirname(pred_csv), exist_ok=True)
                run_via_hybrid(
                    iat_col=iat_col,
                    model=model,
                    inputs=inputs,
                    output_json=output_json,
                    pred_csv=pred_csv,
                    num_none_included=num_none_included,
                    vision_embeddings_dir=vision_embeddings_dir,
                    annotations_path=annotations_path,
                    procedures_embs_path=procedures_embs_path,
                    tasks_embs_path=tasks_embs_path,
                    instrument_tracks_path=instrument_tracks_path,
                    seed=seed,
                    uncertainty_calibration=uncertainty_calibration,
                    num_tracks=num_tracks,
                    metric_avg=metric_avg,
                    multiple_instance_training=multiple_instance_training,
                )
                print('-'*50)

            # combine pred_csvs into one
            combined_df = None
            for iat_col, pred_csv in pred_csvs.items():
                df = pd.read_csv(pred_csv)
                df = df[['cvid', 'dialogue', 'procedure', 'procedure_defn', 'task', 'task_defn', 'instrument', 'action', 'tissue', 'pred', 'confidence', 'pred_probs']].rename(
                    columns={
                        'pred': f'{iat_col}_pred',
                        'confidence': f'{iat_col}_conf',
                        'pred_probs': f'{iat_col}_pred_probs',
                    }
                )
                if combined_df is None:
                    combined_df = df
                else:
                    combined_df = combined_df.merge(df, on=['cvid', 'procedure', 'procedure_defn', 'task', 'task_defn', 'dialogue', 'instrument', 'action', 'tissue'], how='outer')
            combined_pred_csv = os.path.join(
                '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/outputs/iat_predictions',
                output_format.format(iat_col='dummy', model=model, inputs=inputs, num_none_included=num_none_included).replace('dummy-', '') + '.csv'
            )
            if uncertainty_calibration is not None:
                combined_pred_csv = combined_pred_csv.replace('.csv', f'-uncertainty_calibration={uncertainty_calibration}.csv')
            combined_df.to_csv(combined_pred_csv, index=False)

if __name__ == "__main__":
    pass
    
    # main_embs(
    #     output_format='{iat_col}-{model}-{inputs}-none={num_none_included}',
    #     metric_avg='macro',
    # )
    
    # main_tracks()
    # main_hybrid(
    #     output_format='{iat_col}-{model}-{inputs}+tracks-none={num_none_included}-num_tracks=1-final',
    #     num_tracks=1,
    #     instrument_tracks_path='/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/outputs/instrument_tracks/instrument_tracks-num_tracks=1.h5',
    # )
    
    # main_hybrid(
    #     output_format='{iat_col}-{model}-{inputs}+tracks-none={num_none_included}-num_tracks=5-final',
    #     num_tracks=5,
    #     instrument_tracks_path='/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/outputs/instrument_tracks/instrument_tracks-num_tracks=5.h5',
    # )
    
    main_hybrid(
        output_format='{iat_col}-{model}-{inputs}+tracks-none={num_none_included}-num_tracks=15',
        num_tracks=15,
        annotations_path='/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/data/iat_predictor_splits/full.csv',
        instrument_tracks_path='/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/outputs/instrument_tracks/instrument_tracks-num_tracks=15.h5',
        metric_avg='macro',
        multiple_instance_training=False,
    )
    
    # main_hybrid(
    #     output_format='{iat_col}-{model}-{inputs}+tracks-none={num_none_included}-num_tracks=30-final',
    #     num_tracks=30,
    #     instrument_tracks_path='/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/outputs/instrument_tracks/instrument_tracks-num_tracks=30.h5',
    # )
    
    # main_hybrid(
    #     output_format='{iat_col}-{model}-{inputs}+tracks-none={num_none_included}-num_tracks=100-final',
    #     num_tracks=100,
    #     instrument_tracks_path='/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/outputs/instrument_tracks/instrument_tracks-num_tracks=100.h5',
    # )
    
    # main_hybrid(
    #     output_format='{iat_col}-{model}-{inputs}+tracks-none={num_none_included}-no_filter_instrument_tracks-final',
    #     num_tracks=400,
    #     instrument_tracks_path='/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/outputs/instrument_tracks/instrument_tracks-no_filter.h5',
    # )
    