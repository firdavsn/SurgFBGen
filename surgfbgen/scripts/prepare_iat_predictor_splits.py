import pandas as pd
import os
import random
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt

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
            # print(f"Warning: Found {len(procedures_tmp_df)} procedures for case {case}.")
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
            # print(f"Warning: Found {len(tasks_tmp_df)} tasks for case {case} at clip start sec {clip_start_sec}.")
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
    
    extractions_df['instrument-extraction'] = extractions_df['instrument']
    extractions_df['action-extraction'] = extractions_df['action']
    extractions_df['tissue-extraction'] = extractions_df['tissue']
    
    extractions_df = extractions_df.drop(columns=[
        'instrument', 'action', 'tissue',
    ])
    extractions_df = extractions_df.rename(columns={
        'instrument-cluster': 'instrument',
        'action-cluster': 'action',
        'tissue-cluster': 'tissue',
    })
    
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

def find_elbow_point(sorted_counts: np.ndarray, ignore_first_n_points: int = 1) -> int:
    """
    Finds the elbow point in a sorted array of counts using the Kneedle algorithm.
    This version is modified to be more robust for sharp "L-shaped" curves.

    Args:
        sorted_counts: A numpy array of class counts, sorted in descending order.
        ignore_first_n_points: How many of the first (highest) points to ignore
                               when defining the reference line. This helps avoid
                               the influence of initial steep drops.

    Returns:
        The count value at the determined elbow point.
    """
    # We need at least 3 points to form a curve (start, middle, end)
    if len(sorted_counts) < 3:
        return sorted_counts[-1] if len(sorted_counts) > 0 else 0

    # Create a view of the array that skips the first N points
    search_counts = sorted_counts[ignore_first_n_points:]
    
    if len(search_counts) < 2: # Need at least a start and end point for the line
        return sorted_counts[-1]

    # 1. Normalize the data (indices and counts) for the search area
    n_points = len(search_counts)
    all_coords = np.vstack((range(n_points), search_counts)).T
    
    # Normalize to the range [0, 1]
    normalized_coords = all_coords.copy().astype(float)
    x_range = all_coords[:, 0].max() - all_coords[:, 0].min()
    y_range = all_coords[:, 1].max() - all_coords[:, 1].min()
    
    normalized_coords[:, 0] = (all_coords[:, 0] - all_coords[:, 0].min()) / (x_range or 1)
    normalized_coords[:, 1] = (all_coords[:, 1] - all_coords[:, 1].min()) / (y_range or 1)

    # 2. Get the line connecting the first and last points of the search area
    first_point = normalized_coords[0]
    last_point = normalized_coords[-1]
    line_vec = last_point - first_point
    
    line_vec_norm_val = np.sqrt(np.sum(line_vec**2))
    if line_vec_norm_val == 0: # Avoid division by zero if start/end are same
        return sorted_counts[-1]
        
    line_vec_norm = line_vec / line_vec_norm_val

    # 3. Find the distance of all points to the line
    vec_from_first = normalized_coords - first_point
    scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))

    # 4. The point with the maximum distance is the elbow
    elbow_idx_relative = np.argmax(dist_to_line)
    
    # Convert back to an index in the original `sorted_counts` array
    elbow_idx_absolute = elbow_idx_relative + ignore_first_n_points
    
    return sorted_counts[elbow_idx_absolute]

def visualize_and_find_thresholds(extractions_df: pd.DataFrame):
    """
    Visualizes class counts and deterministically finds the elbow threshold for each.
    """
    # Prepare data for plotting and analysis
    data_to_plot = {}
    class_counts = {
        'instrument': extractions_df['instrument'].value_counts().to_dict(),
        'action': extractions_df['action'].value_counts().to_dict(),
        'tissue': extractions_df['tissue'].value_counts().to_dict(),
    }
    thresholds = {}
    for cat in ['instrument', 'action', 'tissue']:
        data_to_plot[cat] = {}
        
        counts = class_counts[cat]
        if 'NONE' in counts:
            data_to_plot[cat]['NONE-count'] = counts['NONE']
            del counts['NONE']
        
        # Sort by count for a clean plot and for the elbow algorithm
        sorted_items = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        names = [item[0] for item in sorted_items]
        values = np.array([item[1] for item in sorted_items])
        
        # Find the elbow threshold using the improved method
        threshold = find_elbow_point(values, ignore_first_n_points=2)
        thresholds[cat] = threshold
        
        data_to_plot[cat].update({'names': names, 'values': values})

    # --- Plotting ---
    plt.figure(figsize=(16, 8))
    colors = {'instrument': 'C0', 'action': 'C1', 'tissue': 'C2'}
    labels = {
        'instrument': "Instrument", 
        'action': "Action", 
        'tissue': "Tissue"
    }

    for cat, data in data_to_plot.items():
        # Plot the curve
        plt.plot(data['names'], data['values'], marker='o', label=labels[cat] + f" (# NONE = {data_to_plot[cat]['NONE-count']})", color=colors[cat])
        
        # Plot a horizontal line at the determined threshold
        plt.axhline(y=thresholds[cat], color=colors[cat], linestyle='--', 
                    label=f'{labels[cat]} Threshold = {thresholds[cat]}')

    print("--- Determined Thresholds ---")
    print(f"Instrument Threshold: {thresholds['instrument']}")
    print(f"Action Threshold:     {thresholds['action']}")
    print(f"Tissue Threshold:     {thresholds['tissue']}")
    print("-------------------------------------------")

    plt.title('IAT Class Counts with Determined Elbow Thresholds')
    plt.xlabel('Classes (Sorted by Frequency)')
    plt.ylabel('Counts')
    plt.xticks(rotation=270)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(os.environ['REPO_DIR'], 'data', 'iat_predictor_splits', 'IAT_class_counts_with_thresholds.png'))
    plt.close()
    
    return thresholds

def ensure_valid_actionable_fb(extractions_df: pd.DataFrame) -> pd.DataFrame:
    df = extractions_df.copy()
    df = df[(df['instrument'] != 'NONE') | (df['action'] != 'NONE') | (df['tissue'] != 'NONE')]
    return df
        
def main():
    # Load dataframes
    extractions_path = os.path.join(os.environ['REPO_DIR'], 'data', 'iat', 'extractions_df.csv')
    procedures_path = os.path.join(os.environ['REPO_DIR'], 'data', 'iat', 'procedures_df.csv')
    tasks_path = os.path.join(os.environ['REPO_DIR'], 'data', 'iat', 'tasks_df.csv')
    extractions_df = pd.read_csv(extractions_path)
    procedures_df = pd.read_csv(procedures_path)
    tasks_df = pd.read_csv(tasks_path)

    # Set procedure and task in extractions_df
    extractions_df = set_procedure_and_task(extractions_df, procedures_df, tasks_df)

    # Load mappings
    instrument_mappings_path = os.path.join(os.environ['REPO_DIR'], 'data', 'iat', 'instrument_clusters-mapping.json')
    action_mappings_path = os.path.join(os.environ['REPO_DIR'], 'data', 'iat', 'action_clusters-mapping.json')
    tissue_mappings_path = os.path.join(os.environ['REPO_DIR'], 'data', 'iat', 'tissue_clusters-mapping.json')
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
    
    extractions_df.to_csv(os.path.join(os.environ['REPO_DIR'], 'data', 'iat_predictor_splits', 'clusters_df.csv'), index=False)
    print(f"Saved extractions_df with clusters to {os.path.join(os.environ['REPO_DIR'], 'data', 'iat_predictor_splits', 'clusters_df.csv')}")
    
    # Remove IAT classes that are too few in the count
    thresholds = visualize_and_find_thresholds(extractions_df)
    # thresholds = {'instrument': 30, 'action': 25, 'tissue': 24}
    for cat in ['instrument', 'action', 'tissue']:
        threshold = thresholds[cat]
        extractions_df = extractions_df[extractions_df[cat].map(lambda x: x in extractions_df[cat].value_counts()[extractions_df[cat].value_counts() >= threshold].index)]
    
    # Filter out non-actionable feedback
    # extractions_df = ensure_valid_actionable_fb(extractions_df)
    # print(f"Counts: {extractions_df['instrument'].value_counts()}")
    # print(f"Counts: {extractions_df['action'].value_counts()}")
    # print(f"Counts: {extractions_df['tissue'].value_counts()}")

    # Count of IAT with 0-NONE, 1-NONE, 2-NONE
    tmp_df = extractions_df.copy()
    tmp_df['num_none'] = tmp_df[['instrument', 'action', 'tissue']].apply(lambda x: sum([1 for v in x if v == 'NONE']), axis=1)
    print(f"Counts of IAT with 0-NONE: {len(tmp_df[tmp_df['num_none'] == 0])}")
    print(f"Counts of IAT with 1-NONE: {len(tmp_df[tmp_df['num_none'] == 1])}")
    print(f"Counts of IAT with 2-NONE: {len(tmp_df[tmp_df['num_none'] == 2])}")
    print(f"Counts of IAT with 3-NONE: {len(tmp_df[tmp_df['num_none'] == 3])}")
    
    tmp_df = tmp_df[tmp_df['action'] != 'NONE']
    print(f"Count of actionable IATs (action != NONE): {len(tmp_df)} ({len(tmp_df) / len(extractions_df) * 100:.2f}%)")
    
    # print(f"Counts after filtering for actionable: {tmp_df['instrument'].value_counts()}")
    # print(f"Counts after filtering for actionable: {tmp_df['action'].value_counts()}")
    # print(f"Counts after filtering for actionable: {tmp_df['tissue'].value_counts()}")
    
    # Get all feedback that have at least one of the IAT classes
    len_before_filtering = len(extractions_df)
    extractions_df = extractions_df[
        (extractions_df['instrument'] != 'NONE') |
        (extractions_df['action'] != 'NONE') |
        (extractions_df['tissue'] != 'NONE')
    ]
    len_after_filtering = len(extractions_df)
    
    print(f"Instrument NONE count: {len(extractions_df[extractions_df['instrument'] == 'NONE'])}, non-NONE count: {len(extractions_df[extractions_df['instrument'] != 'NONE'])}")
    print(f"Action NONE count: {len(extractions_df[extractions_df['action'] == 'NONE'])}, non-NONE count: {len(extractions_df[extractions_df['action'] != 'NONE'])}")
    print(f"Tissue NONE count: {len(extractions_df[extractions_df['tissue'] == 'NONE'])}, non-NONE count: {len(extractions_df[extractions_df['tissue'] != 'NONE'])}")
    print(f"IAT NONE count: {len_before_filtering - len_after_filtering}. IAT non-NONE count: {len_after_filtering}")
    
    # Get split indices
    num_splits = 5
    split_indices = get_split_indices(extractions_df, num_splits)

    # Create splits
    splits = []
    for i, val_indices in enumerate(split_indices):
        val_df = extractions_df.loc[val_indices].copy()
        train_indices = [idx for idx in extractions_df.index if idx not in val_indices]
        train_df = extractions_df.loc[train_indices].copy()
        full_df = pd.concat([train_df, val_df], ignore_index=True)
        splits.append({'train_df': train_df, 'val_df': val_df, 'full_df': full_df})

    # Save splits to CSV files
    output_dir = os.path.join(os.environ['REPO_DIR'], 'data', 'iat_predictor_splits')
    os.makedirs(output_dir, exist_ok=True)
    for i, split in enumerate(splits):
        train_file_path = os.path.join(output_dir, f'train{i + 1}.csv')
        val_file_path = os.path.join(output_dir, f'val{i + 1}.csv')
        split['train_df'].to_csv(train_file_path, index=False)
        split['val_df'].to_csv(val_file_path, index=False)
        print(f"Saved train split {i + 1} to {train_file_path}")
        print(f"Saved val split {i + 1} to {val_file_path}")
    split['full_df'].to_csv(os.path.join(output_dir, f'full.csv'), index=False)
    print(f"Saved {num_splits} splits to {output_dir}")
        
if __name__ == "__main__":
    main()