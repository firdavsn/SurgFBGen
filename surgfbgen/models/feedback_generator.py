from pydantic import BaseModel
import pandas as pd
import os
import json
from typing import Tuple, List, Dict, Union
import numpy as np
import cv2
from tqdm import tqdm

from surgfbgen.prompts.chatllm_interface import ChatLLMInterface
from surgfbgen.prompts import prompt_library
from surgfbgen.prompts.cli import format_prompt

PROMPT_NAMES = {
    # (Frames, IAT)
    (True, True): "generate_feedback_from_frames_and_iat",
    (True, False): "generate_feedback_from_frames",
    (False, True): "generate_feedback_from_iat",
    (False, False): "generate_feedback_no_input",
}

IAT_CLASS_DEFINITIONS_PATH = os.path.join(os.environ['REPO_DIRECTORY'], 'data', 'iat', 'iat_class_definitions.json')
with open(IAT_CLASS_DEFINITIONS_PATH, 'r') as f:
    IAT_CLASS_DEFINITIONS = json.load(f)


IAT_CLASS_DEFINITIONS['instrument']['UNCERTAIN'] = "The instrument is uncertain or not clearly identifiable."
IAT_CLASS_DEFINITIONS['action']['UNCERTAIN'] = "The action is uncertain or not clearly identifiable."
IAT_CLASS_DEFINITIONS['tissue']['UNCERTAIN'] = "The tissue is uncertain or not clearly identifiable."

class FeedbackGeneratorConfig(BaseModel):
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
    def __init__(
        self,
        config: FeedbackGeneratorConfig,
        api_key: str,   # either OpenAI or Gemini model. will be inferred from the model name
    ):
        self.config = config
        self.chatllm_interface = ChatLLMInterface(
            model_name=config.chatllm_name,
            api_key=api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    
    def _format_iat_triplets(
        self,
        instrument: str,
        action: str,
        tissue: str,
    ) -> str:
        return f"(Instrument: {instrument}, Action: {action}, Tissue: {tissue})"
    
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
    
    def _get_and_format_class_definitions(
        self,
        instrument: str,
        action: str,
        tissue: str,
    ) -> str:
        class_defintions = {}
        if instrument not in [None, 'NONE', '']:
            class_defintions['Instrument'] = (instrument, IAT_CLASS_DEFINITIONS['instrument'][instrument])
        if action not in [None, 'NONE', '']:
            
            class_defintions['Action'] = (action, IAT_CLASS_DEFINITIONS['action'][action])
        if tissue not in [None, 'NONE', '']:
            class_defintions['Tissue'] = (tissue, IAT_CLASS_DEFINITIONS['tissue'][tissue])
        return json.dumps(class_defintions, indent=4)
    
    def _get_and_reference_examples(
        self, 
        row: pd.Series,
        data_df: pd.DataFrame,
    ):
        df = data_df.copy()
        df = df[df.index != row.name]  # Exclude the current row
        num_examples = self.config.num_reference_examples
        examples_granularity = self.config.reference_examples_granularity
        
        if examples_granularity == 'all':
            # Get all examples from the dataset
            examples = df.sample(n=num_examples, random_state=row.name)
        elif examples_granularity == 'procedure':
            examples = df[df['procedure'] == row['procedure']].sample(n=num_examples, random_state=row.name)
        elif examples_granularity == 'task':
            examples = df[df['task'] == row['task']].sample(n=num_examples, random_state=row.name)
        elif examples_granularity == 'iat':
            examples = df[
                (df['instrument'] == row['instrument']) &
                (df['action'] == row['action']) &
                (df['tissue'] == row['tissue'])
            ].sample(n=num_examples, random_state=row.name)
        else:
            raise ValueError(f"Unknown reference examples granularity: {examples_granularity}")
        
        examples_str = ""
        for _, ex_row in examples.iterrows():
            iat_triplet = self._format_iat_triplets(
                ex_row['instrument'],
                ex_row['action'],
                ex_row['tissue']
            )
            feedback = ex_row['dialogue']
            examples_str += f"  - {iat_triplet} -> '{feedback}'\n"
        return examples_str.strip()
    
    def _generate_feedback(
        self,
        row: pd.Series,
        data_df: pd.DataFrame
    ) -> str:
        prompt_args = {}
        if self.config.input_iat_triplet:
            iat_str = self._format_iat_triplets(
                row['instrument'],
                row['action'],
                row['tissue']
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
                row['instrument'],
                row['action'],
                row['tissue']
            )
        else:
            class_definitions = {}
        
        reference_examples = self._get_and_reference_examples(
            row,
            data_df=data_df
        )
        
        prompt_args = {
            'iat_triplet': iat_str,
            'frames': frames,
            'procedure': procedure,
            'procedure_defn': procedure_defn,
            'task': task,
            'task_defn': task_defn,
            'class_definitions': class_definitions,
            'reference_examples': reference_examples,
            'video_frames': ''
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
    ):
        # data_df.columns = [dialogue, cvid, instrument, action, tissue, procedure, procedure_defn, task, task_defn, (feedback)]
        
        output_rows = []
        for i in tqdm(range(len(data_df)), desc="Generating feedback"):
            row = data_df.iloc[i]
            row_dict = row.to_dict()
            if not override_existing and 'feedback' in row_dict and not pd.isna(row_dict['feedback']):
                print("Warning: 'feedback' column already exists in the input DataFrame. Skipping.")
                output_rows.append(row_dict)
                continue
            feedback = self._generate_feedback(row, data_df)
            row_dict['feedback'] = feedback
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