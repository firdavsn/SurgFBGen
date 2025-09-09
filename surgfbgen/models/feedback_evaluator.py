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

PROMPT_NAME = 'judge_feedback'

IAT_CLASS_DEFINITIONS_PATH = os.path.join(os.environ['REPO_DIRECTORY'], 'data', 'iat', 'iat_class_definitions.json')
with open(IAT_CLASS_DEFINITIONS_PATH, 'r') as f:
    IAT_CLASS_DEFINITIONS = json.load(f)


class FeedbackEvaluatorConfig(BaseModel):
    chatllm_name: str
    temperature: float = 0.2
    max_tokens: int = 10_000
    
class FeedbackEvaluator:
    def __init__(
        self,
        config: FeedbackEvaluatorConfig,
        api_key: str,   # either OpenAI or Gemini model. will be inferred from the model name
    ):
        self.config = config
        self.chatllm_interface = ChatLLMInterface(
            model_name=config.chatllm_name,
            api_key=api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    
    def _generate_score(
        self,
        row: pd.Series,
        data_df: pd.DataFrame,
        gt_fb_col: str = 'dialogue',
        gen_fb_col: str = 'feedback'
    ) -> str:
        prompt_args = {}
        dialogue = row[gt_fb_col]
        feedback = row[gen_fb_col]
        prompt_args = {
            'gen_fb': feedback,
            'gt_fb': dialogue,
        }
        prompt = format_prompt(
            name=PROMPT_NAME,
            kwargs=prompt_args
        )
        response = self.chatllm_interface.generate(
            user_prompt=prompt
        )
        return response.strip()
    
    def generate_all_scores(
        self,
        data_df: pd.DataFrame,
        gt_fb_col: str = 'dialogue',
        gen_fb_col: str = 'feedback'
    ):
        # data_df.columns = [dialogue, cvid, instrument, action, tissue, procedure, procedure_defn, task, task_defn, ]
        
        output_rows = []
        for i in tqdm(range(len(data_df)), desc="Evaluating feedback"):
            row = data_df.iloc[i]
            score = self._generate_score(row, data_df, gt_fb_col, gen_fb_col)
            row_dict = row.to_dict()
            row_dict['score'] = score
            output_rows.append(row_dict)
        
        output_df = pd.DataFrame(output_rows)
        return output_df

