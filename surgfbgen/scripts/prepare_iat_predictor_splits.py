import pandas as pd
import os
import random
from typing import Dict, List, Optional

import surgfbgen.config.environment

def set_procedure_and_task(
    extractions_df: pd.DataFrame, procedures_df: pd.DataFrame, tasks_df: pd.DataFrame
) -> pd.DataFrame:
    """Set procedure and task for each extraction based on the case and clip start time."""

    extractions_df['clip_start_sec'] = extractions_df['cvid'].apply(lambda x: sum([a*b for a,b in zip(map(int, x.split('_')[-1][:-4].split('-')), [3600, 60, 1])]))
    extractions_df['procedure'] = None
    extractions_df['procedure_defn'] = None
    extractions_df['task'] = None
    extractions_df['task_defn'] = None
    for i in range(len(extractions_df)):
        row = extractions_df.iloc[i]
        case = row['case']
        clip_start_sec = row['clip_start_sec']
        
        procedures_tmp_df = procedures_df[procedures_df['case_id'] == case]
        if len(procedures_tmp_df) != 1:
            print(f"Warning: Found {len(procedures_tmp_df)} procedures for case {case}.")
            procedure = None
            procedure_defn = None
        else:
            procedure = procedures_tmp_df['procedure'].values[0] if not procedures_tmp_df.empty else None
            procedure_defn = procedures_tmp_df['procedure_defn'].values[0] if not procedures_tmp_df.empty else None
        
        tasks_tmp_df = tasks_df[tasks_df['case_id'] == case]
        tasks_tmp_df = tasks_tmp_df[(tasks_tmp_df['start_secs'] <= clip_start_sec) & (tasks_tmp_df['end_secs'] > clip_start_sec)]
        tasks_tmp_df['task'] = tasks_tmp_df['task'].apply(lambda x: x.strip() if isinstance(x, str) else x)
        tasks_tmp_df = tasks_tmp_df.drop_duplicates(subset=['task']).dropna(subset=['task'])
        if len(tasks_tmp_df) != 1:
            print(f"Warning: Found {len(tasks_tmp_df)} tasks for case {case} at clip start sec {clip_start_sec}.")
            task = None
            task_defn = None
        else:
            task = tasks_tmp_df['task'].values[0] if not tasks_tmp_df.empty else None
            task_defn = tasks_tmp_df['task_defn'].values[0] if not tasks_tmp_df.empty else None

        extractions_df.at[i, 'procedure'] = procedure
        extractions_df.at[i, 'procedure_defn'] = procedure_defn
        extractions_df.at[i, 'task'] = task
        extractions_df.at[i, 'task_defn'] = task_defn
        
    extractions_df = extractions_df.drop(columns=['clip_start_sec'])
    
    return extractions_df

def map_extractions_to_clusters(
    extractions_df: pd.DataFrame, 
    instrument_mappings: Dict[str, str],
    action_mappings: Dict[str, str],
    tissue_mappings: Dict[str, str]
) -> pd.DataFrame:
    """Map extractions to their respective clusters."""
    
    extractions_df['instrument-cluster'] = extractions_df['instrument'].map(instrument_mappings).fillna("NONE")
    extractions_df['action-cluster'] = extractions_df['action'].map(action_mappings).fillna("NONE")
    extractions_df['tissue-cluster'] = extractions_df['tissue'].map(tissue_mappings).fillna("NONE")
    
    return extractions_df

def get_split_indices(df: pd.DataFrame, num_splits: int, shuffle: bool = True) -> list:
    """Get indices for splitting the DataFrame into num_splits, with optional shuffling."""
    idx = df.index.to_list()
    if shuffle:
        random.shuffle(idx)
    split_size = len(idx) // num_splits
    indices = []
    for i in range(num_splits):
        start_idx = i * split_size
        end_idx = start_idx + split_size if i < num_splits - 1 else len(idx)
        indices.append(idx[start_idx:end_idx])
    return indices

def set_correct_cols(extractions_df: pd.DataFrame) -> pd.DataFrame:
    extractions_df = extractions_df.drop(columns=[
        'instrument', 'action', 'tissue',
    ])
    extractions_df = extractions_df.rename(columns={
        'instrument-cluster': 'instrument',
        'action-cluster': 'action',
        'tissue-cluster': 'tissue',
    })
    return extractions_df

def main():
    # Load dataframes
    extractions_path = os.path.join(os.environ['REPO_DIRECTORY'], 'data', 'iat', 'extractions_df.csv')
    procedures_path = os.path.join(os.environ['REPO_DIRECTORY'], 'data', 'iat', 'procedures_df.csv')
    tasks_path = os.path.join(os.environ['REPO_DIRECTORY'], 'data', 'iat', 'tasks_df.csv')
    extractions_df = pd.read_csv(extractions_path)
    procedures_df = pd.read_csv(procedures_path)
    tasks_df = pd.read_csv(tasks_path)

    # Set procedure and task in extractions_df
    extractions_df = set_procedure_and_task(extractions_df, procedures_df, tasks_df)

    # Load mappings
    instrument_mappings_path = os.path.join(os.environ['REPO_DIRECTORY'], 'data', 'iat', 'o3_instrument_clusters-mapping.json')
    action_mappings_path = os.path.join(os.environ['REPO_DIRECTORY'], 'data', 'iat', 'o3_action_clusters-mapping.json')
    tissue_mappings_path = os.path.join(os.environ['REPO_DIRECTORY'], 'data', 'iat', 'o3_tissue_clusters-mapping.json')
    instrument_mappings = pd.read_json(instrument_mappings_path, typ='series').to_dict()
    action_mappings = pd.read_json(action_mappings_path, typ='series').to_dict()
    tissue_mappings = pd.read_json(tissue_mappings_path, typ='series').to_dict()

    # Map extractions to clusters
    extractions_df = map_extractions_to_clusters(
        extractions_df, 
        instrument_mappings=instrument_mappings,
        action_mappings=action_mappings,
        tissue_mappings=tissue_mappings
    )

    # Set correct columns
    extractions_df = set_correct_cols(extractions_df)

    # Get split indices
    num_splits = 5
    split_indices = get_split_indices(extractions_df, num_splits)

    # Create splits
    splits = []
    for i, val_indices in enumerate(split_indices):
        val_df = extractions_df.loc[val_indices].copy()
        train_indices = [idx for idx in extractions_df.index if idx not in val_indices]
        train_df = extractions_df.loc[train_indices].copy()
        splits.append({'train_df': train_df, 'val_df': val_df})

    # Save splits to CSV files
    output_dir = os.path.join(os.environ['REPO_DIRECTORY'], 'data', 'iat_predictor_splits')
    os.makedirs(output_dir, exist_ok=True)
    for i, split in enumerate(splits):
        train_file_path = os.path.join(output_dir, f'train{i + 1}.csv')
        val_file_path = os.path.join(output_dir, f'val{i + 1}.csv')
        split['train_df'].to_csv(train_file_path, index=False)
        split['val_df'].to_csv(val_file_path, index=False)
        print(f"Saved train split {i + 1} to {train_file_path}")
        print(f"Saved val split {i + 1} to {val_file_path}")
        
if __name__ == "__main__":
    main()