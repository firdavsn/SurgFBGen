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
    vis_embs_avg = np.array(df[f'{model_name}_vis_embs'].values.tolist()).mean(axis=1)
    
    emb_col = ''
    if model_name == 'surgvlp': emb_col = 'SurgVLP'
    elif model_name == 'hecvl': emb_col = 'HecVL'
    elif model_name == 'peskavlp': emb_col = 'PeskaVLP'
    elif model_name == 'pe224' or model_name == 'pe336' or model_name == 'pe448': emb_col = 'MedEmbed_small'
    elif model_name == 'videomae_urology' or model_name == 'videomae_cholect45': emb_col = 'MedEmbed_small'
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
    
    
    df['vis_embs_avg'] = list(vis_embs_avg)
    df['procedure_embs'] = list(procedure_embs) if procedures_df is not None else None
    df['task_embs'] = list(task_embs) if tasks_df is not None else None
    df['label'] = list(labels)
    df = df.reset_index(drop=True)
    
    if procedures_df is not None and tasks_df is not None:
        embeddings_comb = np.concatenate([vis_embs_avg, procedure_embs, task_embs], axis=1)
    elif procedures_df is not None:
        embeddings_comb = np.concatenate([vis_embs_avg, procedure_embs], axis=1)
    elif tasks_df is not None:
        embeddings_comb = np.concatenate([vis_embs_avg, task_embs], axis=1)
    else:
        embeddings_comb = vis_embs_avg
    df['embedding'] = list(embeddings_comb)
    
    # print emb dims
    print(f"Vision Embedding Dimension: {vis_embs_avg.shape[1]}")
    print(f"Procedure Embedding Dimension: {procedure_embs.shape[1]}")
    print(f"Task Embedding Dimension: {task_embs.shape[1]}")
    print(f"Combined Embedding Dimension: {embeddings_comb.shape[1]}")
    
    return df

def run_via_embs(
    iat_col: str,   # instrument, action, or tissue
    model: str, # peskavlp, surgvlp, hecvl, pe224, pe336, pe448, videomae_urology, videomae_cholect45
    inputs: str, # vision, vision+procedure, vision+task, vision
    output_json: str,
    pred_csv: str = None,
    num_none_included: int = 100,
    vision_embeddings_dir: str = '~/surgery/surgical_fb_generation/SurgFBGen/outputs/embeddings/vision',
    annotations_path: str = '~/surgery/surgical_fb_generation/SurgFBGen/data/iat_predictor_splits/full.csv',
    procedures_embs_path: str = '~/surgery/surgical_fb_generation/SurgFBGen/data/iat/procedures_embs_df.parquet',
    tasks_embs_path: str = '~/surgery/surgical_fb_generation/SurgFBGen/data/iat/tasks_embs_df.parquet',
    seed: int = 0,
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
    metrics, pred_df = evaluate_via_embs(processed_df, metric_avg='weighted', num_folds=5, hidden_layer_sizes=(64, 32, 16), seed=seed)

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
                _, (hidden_state, _) = model.lstm(batch_tracks_device)
                track_summary = hidden_state[-1]
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

# evaluate_via_hybrid = evaluate_via_hybrid_original

def run_via_hybrid(
    iat_col: str,
    model: str, 
    inputs: str,
    output_json: str,
    pred_csv: str = None,
    num_none_included: int = 100,
    vision_embeddings_dir: str = '~/surgery/surgical_fb_generation/SurgFBGen/outputs/embeddings/vision',
    annotations_path: str = '~/surgery/surgical_fb_generation/SurgFBGen/data/iat_predictor_splits/full.csv',
    procedures_embs_path: str = '~/surgery/surgical_fb_generation/SurgFBGen/data/iat/procedures_embs_df.parquet',
    tasks_embs_path: str = '~/surgery/surgical_fb_generation/SurgFBGen/data/iat/tasks_embs_df.parquet',
    instrument_tracks_path: str = '~/surgery/surgical_fb_generation/SurgFBGen/outputs/instrument_tracks/instrument_tracks.h5',
    seed: int = 0,
    num_tracks: int = 15,
    uncertainty_calibration: str = None,  # None or 'platt'
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
    
    h5 = h5py.File(instrument_tracks_path, 'r')
    processed_df['tracks'] = processed_df['cvid'].apply(lambda x: standardize_track_shape(h5[x][:][0], target_dots=num_tracks) if x in h5 else None)
    h5.close()
    print(processed_df['tracks'].iloc[0].shape) 
    
    processed_df.dropna(subset=['tracks', 'embedding'], inplace=True)
    processed_df = processed_df.reset_index(drop=True)
    print(f"Number of samples after loading embeddings and tracks: {len(processed_df)}")

    metrics, pred_df = evaluate_via_hybrid(
        processed_df, 
        metric_avg='weighted', 
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
    vision_embeddings_dir: str = '~/surgery/surgical_fb_generation/SurgFBGen/outputs/embeddings/vision',
    annotations_path: str = '~/surgery/surgical_fb_generation/SurgFBGen/data/iat_predictor_splits/full.csv',
    procedures_embs_path: str = '~/surgery/surgical_fb_generation/SurgFBGen/data/iat/procedures_embs_df.parquet',
    tasks_embs_path: str = '~/surgery/surgical_fb_generation/SurgFBGen/data/iat/tasks_embs_df.parquet',
    seed: int = 0,
):
    for model in [
        'peskavlp', 
        'surgvlp', 
    ]:
        for inputs in [
            'vision',
            'vision+procedure',
            'vision+procedure+task',
        ]:
            pred_csvs = {}
            for iat_col in ['instrument', 'action', 'tissue']:
                output_json = os.path.join(
                    '~/surgery/surgical_fb_generation/SurgFBGen/outputs/iat_metrics',
                    output_format.format(iat_col=iat_col, model=model, inputs=inputs, num_none_included=num_none_included) + '.json'    
                )
                pred_csv = os.path.join(
                    '~/surgery/surgical_fb_generation/SurgFBGen/outputs/iat_predictions',
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
                '~/surgery/surgical_fb_generation/SurgFBGen/outputs/iat_predictions',
                output_format.format(iat_col='dummy', model=model, inputs=inputs, num_none_included=num_none_included).replace('dummy-', '') + '.csv'
            )
            combined_df.to_csv(combined_pred_csv, index=False)

def main_hybrid(
    output_format: str = '{iat_col}-{model}-{inputs}+tracks-none={num_none_included}',
    num_none_included: int = 100,
    vision_embeddings_dir: str = '~/surgery/surgical_fb_generation/SurgFBGen/outputs/embeddings/vision',
    annotations_path: str = '~/surgery/surgical_fb_generation/SurgFBGen/data/iat_predictor_splits/full.csv',
    procedures_embs_path: str = '~/surgery/surgical_fb_generation/SurgFBGen/data/iat/procedures_embs_df.parquet',
    tasks_embs_path: str = '~/surgery/surgical_fb_generation/SurgFBGen/data/iat/tasks_embs_df.parquet',
    instrument_tracks_path: str = '~/surgery/surgical_fb_generation/SurgFBGen/outputs/instrument_tracks/instrument_tracks-num_tracks=15.h5',
    seed: int = 0,
    num_tracks: int = 15,
    uncertainty_calibration: str = None,  # None or 'platt'
):
    for model in [
        'peskavlp', 
        'surgvlp', 
    ]:
        for inputs in [
            'vision',
            'vision+procedure',
            'vision+procedure+task',
        ]:
            pred_csvs = {}
            for iat_col in ['instrument', 'action', 'tissue']:
                output_json = os.path.join(
                    '~/surgery/surgical_fb_generation/SurgFBGen/outputs/iat_metrics',
                    output_format.format(iat_col=iat_col, model=model, inputs=inputs, num_none_included=num_none_included) + '.json'
                )
                pred_csv = os.path.join(
                    '~/surgery/surgical_fb_generation/SurgFBGen/outputs/iat_predictions',
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
                '~/surgery/surgical_fb_generation/SurgFBGen/outputs/iat_predictions',
                output_format.format(iat_col='dummy', model=model, inputs=inputs, num_none_included=num_none_included).replace('dummy-', '') + '.csv'
            )
            if uncertainty_calibration is not None:
                combined_pred_csv = combined_pred_csv.replace('.csv', f'-uncertainty_calibration={uncertainty_calibration}.csv')
            combined_df.to_csv(combined_pred_csv, index=False)

if __name__ == "__main__":
    pass
    
    # main_embs(
    #     output_format='{iat_col}-{model}-{inputs}-none={num_none_included}-final-with_conf',
    # )
    # main_tracks()
    # main_hybrid(
    #     output_format='{iat_col}-{model}-{inputs}+tracks-none={num_none_included}-num_tracks=1-final',
    #     num_tracks=1,
    #     instrument_tracks_path='~/surgery/surgical_fb_generation/SurgFBGen/outputs/instrument_tracks/instrument_tracks-num_tracks=1.h5',
    # )
    
    # main_hybrid(
    #     output_format='{iat_col}-{model}-{inputs}+tracks-none={num_none_included}-num_tracks=5-final',
    #     num_tracks=5,
    #     instrument_tracks_path='~/surgery/surgical_fb_generation/SurgFBGen/outputs/instrument_tracks/instrument_tracks-num_tracks=5.h5',
    # )
    
    # main_hybrid(
    #     output_format='{iat_col}-{model}-{inputs}+tracks-none={num_none_included}-num_tracks=15-final',
    #     num_tracks=15,
    #     annotations_path='~/surgery/surgical_fb_generation/SurgFBGen/data/iat_predictor_splits/full.csv',
    #     instrument_tracks_path='~/surgery/surgical_fb_generation/SurgFBGen/outputs/instrument_tracks/instrument_tracks-num_tracks=15.h5',
    # )
    
    # main_hybrid(
    #     output_format='{iat_col}-{model}-{inputs}+tracks-none={num_none_included}-num_tracks=30-final',
    #     num_tracks=30,
    #     instrument_tracks_path='~/surgery/surgical_fb_generation/SurgFBGen/outputs/instrument_tracks/instrument_tracks-num_tracks=30.h5',
    # )
    
    # main_hybrid(
    #     output_format='{iat_col}-{model}-{inputs}+tracks-none={num_none_included}-num_tracks=100-final',
    #     num_tracks=100,
    #     instrument_tracks_path='~/surgery/surgical_fb_generation/SurgFBGen/outputs/instrument_tracks/instrument_tracks-num_tracks=100.h5',
    # )
    
    # main_hybrid(
    #     output_format='{iat_col}-{model}-{inputs}+tracks-none={num_none_included}-no_filter_instrument_tracks-final',
    #     num_tracks=400,
    #     instrument_tracks_path='~/surgery/surgical_fb_generation/SurgFBGen/outputs/instrument_tracks/instrument_tracks-no_filter.h5',
    # )