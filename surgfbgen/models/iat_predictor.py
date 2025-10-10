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

IAT_CLASS_DEFINITIONS_PATH = os.path.join(os.environ['REPO_DIRECTORY'], 'data', 'iat', 'iat_class_definitions.json')
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