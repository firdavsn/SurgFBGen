import os
import json
import pandas as pd
from pydantic import BaseModel
from typing import Tuple, List, Dict, Union
import numpy as np
import cv2
from tqdm import tqdm

from surgfbgen.prompts.chatllm_interface import ChatLLMInterface
from surgfbgen.prompts import prompt_library
from surgfbgen.prompts.cli import format_prompt

# A dictionary mapping the configuration of (has_frames, has_iat) to a prompt template name
PROMPT_NAMES = {
    # (Frames, IAT)
    (True, True): "generate_feedback_from_frames_and_iat",
    (True, False): "generate_feedback_from_frames",
    (False, True): "generate_feedback_from_iat",
    (False, False): "generate_feedback_no_input",
}

# Load IAT class definitions from the repository
IAT_CLASS_DEFINITIONS_PATH = os.path.join(os.environ['REPO_DIR'], 'data', 'iat', 'iat_class_definitions.json')
with open(IAT_CLASS_DEFINITIONS_PATH, 'r') as f:
    IAT_CLASS_DEFINITIONS = json.load(f)

# Add definitions for UNCERTAIN, as these might be in the prediction data
IAT_CLASS_DEFINITIONS['instrument']['UNCERTAIN'] = "The instrument is uncertain or not clearly identifiable."
IAT_CLASS_DEFINITIONS['action']['UNCERTAIN'] = "The action is uncertain or not clearly identifiable."
IAT_CLASS_DEFINITIONS['tissue']['UNCERTAIN'] = "The tissue is uncertain or not clearly identifiable."

class FeedbackGeneratorConfig(BaseModel):
    """Configuration settings for the FeedbackGenerator."""
    input_frames: bool = True
    input_iat_triplet: bool = True
    input_class_definitions: bool = True
    
    clips_dir: str
    
    num_reference_examples: int = 10
    reference_examples_granularity: str = 'all' # all, procedure, task, iat
    
    chatllm_name: str
    temperature: float = 0.2
    max_tokens: int = 10_000
    
class FeedbackGenerator:
    """
    Generates surgical feedback using an LLM based on IAT triplets, video frames,
    and contextual definitions.
    """
    def __init__(
        self,
        config: FeedbackGeneratorConfig,
        api_key: str,
        instrument_col: str = 'instrument',
        action_col: str = 'action',
        tissue_col: str = 'tissue',
    ):
        """
        Initializes the FeedbackGenerator.

        Args:
            config: A FeedbackGeneratorConfig object with settings.
            api_key: The API key for the LLM (OpenAI or Gemini).
            instrument_col: The column name in the DataFrame for the instrument data.
            action_col: The column name in the DataFrame for the action data.
            tissue_col: The column name in the DataFrame for the tissue data.
        """
        self.config = config
        self.chatllm_interface = ChatLLMInterface(
            model_name=config.chatllm_name,
            api_key=api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        
        # --- Store the column names ---
        self.instrument_col = instrument_col
        self.action_col = action_col
        self.tissue_col = tissue_col
    
    def _format_iat_triplets(
        self,
        instrument: str,
        action: str,
        tissue: str,
    ) -> str:
        """Formats the IAT values into a consistent string."""
        return f"(Instrument: {instrument}, Action: {action}, Tissue: {tissue})"
    
    def _get_and_format_frames(
        self,
        cvid: str
    ) -> List[np.ndarray]:
        """Loads and samples frames for a given clip ID."""
        clip_path = os.path.join(self.config.clips_dir, cvid)  
        frames = _load_and_sample_video(
            clip_path=clip_path,
            resize_hw=(224, 224),  # Assuming a fixed size for simplicity
            num_frames=10
        )
        frames_list = [frame for frame in frames]
        return frames_list
    
    def _get_and_format_class_definitions(
        self,
        instrument: str,
        action: str,
        tissue: str,
    ) -> str:
        """Retrieves and formats the definitions for the given IAT classes."""
        class_defintions = {}
        if instrument not in [None, 'NONE', '']:
            class_defintions['Instrument'] = (instrument, IAT_CLASS_DEFINITIONS['instrument'].get(instrument, "No definition available."))
        if action not in [None, 'NONE', '']:
            class_defintions['Action'] = (action, IAT_CLASS_DEFINITIONS['action'].get(action, "No definition available."))
        if tissue not in [None, 'NONE', '']:
            class_defintions['Tissue'] = (tissue, IAT_CLASS_DEFINITIONS['tissue'].get(tissue, "No definition available."))
        return json.dumps(class_defintions, indent=4)
    
    def _get_and_reference_examples(
        self, 
        row: pd.Series,
        data_df: pd.DataFrame,
    ) -> str:
        """Finds and formats reference examples based on the configuration."""
        df = data_df.copy()
        df = df[df.index != row.name]  # Exclude the current row
        
        if df.empty:
            return "No other examples available."

        num_examples = min(self.config.num_reference_examples, len(df))
        if num_examples == 0:
            return "No other examples available."
            
        examples_granularity = self.config.reference_examples_granularity
        
        # --- Use the stored column names ---
        inst_col, act_col, tis_col = self.instrument_col, self.action_col, self.tissue_col
        
        if examples_granularity == 'all':
            examples = df.sample(n=num_examples, random_state=row.name if row.name < 2**32 else 42) # handle large row.name for seed
        elif examples_granularity == 'procedure':
            examples = df[df['procedure'] == row['procedure']].sample(n=num_examples, random_state=row.name, replace=True)
        elif examples_granularity == 'task':
            examples = df[df['task'] == row['task']].sample(n=num_examples, random_state=row.name, replace=True)
        elif examples_granularity == 'iat':
            examples = df[
                (df[inst_col] == row[inst_col]) &
                (df[act_col] == row[act_col]) &
                (df[tis_col] == row[tis_col])
            ].sample(n=num_examples, random_state=row.name, replace=True)
        else:
            raise ValueError(f"Unknown reference examples granularity: {examples_granularity}")
        
        examples_str = ""
        for _, ex_row in examples.iterrows():
            iat_triplet = self._format_iat_triplets(
                ex_row[inst_col],
                ex_row[act_col],
                ex_row[tis_col]
            )
            feedback = ex_row['dialogue']
            examples_str += f"  - {iat_triplet} -> '{feedback}'\n"
        return examples_str.strip()
    
    def _generate_feedback(
        self,
        row: pd.Series,
        data_df: pd.DataFrame
    ) -> str:
        """Generates feedback for a single row by formatting the prompt and querying the LLM."""
        prompt_args = {}
        
        # --- Use the stored column names ---
        inst_col, act_col, tis_col = self.instrument_col, self.action_col, self.tissue_col
        
        if self.config.input_iat_triplet:
            iat_str = self._format_iat_triplets(
                row[inst_col],
                row[act_col],
                row[tis_col]
            )
        else:
            iat_str = None
        
        if self.config.input_frames:
            frames = self._get_and_format_frames(row['cvid'])
        else:
            frames = None
            
        
        procedure = row['procedure']
        procedure_defn = row['procedure_defn']
        
        task = row['task']
        task_defn = row['task_defn']
        
        if self.config.input_class_definitions:
            class_definitions = self._get_and_format_class_definitions(
                row[inst_col],
                row[act_col],
                row[tis_col]
            )
        else:
            class_definitions = {}
        
        reference_examples = self._get_and_reference_examples(
            row,
            data_df=data_df
        )
        
        prompt_args = {
            'iat_triplet': iat_str,
            'procedure': procedure,
            'procedure_defn': procedure_defn,
            'task': task,
            'task_defn': task_defn,
            'class_definitions': class_definitions,
            'reference_examples': reference_examples,
        }
        prompt_name = PROMPT_NAMES[
            (self.config.input_frames, self.config.input_iat_triplet)
        ]
        prompt = format_prompt(
            name=prompt_name,
            kwargs=prompt_args
        )
        prompt = prompt.split('Video Frames:\n')[0]

        user_prompt = [prompt]
        if self.config.input_frames and frames is not None:
            user_prompt.append('Video Frames:\n')
            for frame in frames:
                user_prompt.append(frame)
        user_prompt.append('\n\nActionable Feedback:\n')
            
        response = self.chatllm_interface.generate(
            user_prompt=user_prompt
        )
        return response.strip()
    
    def generate_all_feedback(
        self,
        data_df: pd.DataFrame, 
        override_existing: bool = False,
    ) -> pd.DataFrame:
        """
        Generates feedback for all rows in a DataFrame.

        Args:
            data_df: The input DataFrame containing columns for 'cvid',
                procedure, task, and the specified IAT columns.
            override_existing: If True, will regenerate feedback even if
                a 'feedback' column already exists.

        Returns:
            A new DataFrame with a 'feedback' column added/updated.
        """
        
        output_rows = []
        for i in tqdm(range(len(data_df)), desc="Generating feedback"):
            row = data_df.iloc[i]
            row_dict = row.to_dict()
            if not override_existing and 'feedback' in row_dict and not pd.isna(row_dict['feedback']):
                print("Warning: 'feedback' column already exists in the input DataFrame. Skipping.")
                output_rows.append(row_dict)
                continue
            
            try:
                feedback = self._generate_feedback(row, data_df)
                row_dict['feedback'] = feedback
            except Exception as e:
                print(f"Error generating feedback for row {i} (cvid: {row.get('cvid')}): {e}")
                row_dict['feedback'] = f"ERROR: {e}"
                
            output_rows.append(row_dict)
        
        output_df = pd.DataFrame(output_rows)
        return output_df

def _load_and_sample_video(clip_path: str, resize_hw: Tuple[int, int], num_frames: int = None) -> np.ndarray:
    """Loads, samples, and resizes video frames."""
    clip_path = os.path.expanduser(clip_path) # Handle '~'
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video: {clip_path}")
    
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video is empty or cannot be read: {clip_path}")

    # Uniformly sample indices
    if num_frames is None:
        num_frames = total_frames
        
    if total_frames <= num_frames:
        indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
    else:
        step = total_frames / float(num_frames)
        indices = [min(int(i * step), total_frames - 1) for i in range(num_frames)]
    
    frames = []
    unique_indices = sorted(list(set(indices)))
    for idx in unique_indices: # Read each unique frame only once
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            H, W = resize_hw
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from video: {clip_path}")

    # Create the final list by selecting from the read frames
    frame_map = {idx: frame for idx, frame in zip(unique_indices, frames)}
    sampled = [frame_map[i] for i in indices]

    return np.stack(sampled, axis=0) # [T,H,W,C]