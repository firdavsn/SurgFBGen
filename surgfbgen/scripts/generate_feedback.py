import os
import json
import pandas as pd

from surgfbgen.models.feedback_generator import FeedbackGenerator, FeedbackGeneratorConfig


clips_dir = '~/surgery/clips_with_wiggle/fb_clips_wiggle'

# full_split_path = '~/surgery/surgical_fb_generation/SurgFBGen/data/iat_predictor_splits/full.csv'
# full_split_path = '~/surgery/surgical_fb_generation/SurgFBGen/data/iat/extractions_df.csv'

full_split_path = '~/surgery/surgical_fb_generation/SurgFBGen/outputs/iat_predictions/peskavlp-vision+procedure+task+tracks-none=100-num_tracks=15-final.csv'
# full_split_path = '~/surgery/surgical_fb_generation/SurgFBGen/outputs/iat_predictions/peskavlp-vision+procedure+task-none=100-final.csv'
# full_split_path = '~/surgery/surgical_fb  _generation/SurgFBGen/outputs/iat_predictions/peskavlp-vision_procedure_task+tracks-none=100-uncertainty_calibration=platt.csv'
full_split_df = pd.read_csv(full_split_path)

# full_split_df['instrument'] = full_split_df['instrument-extraction']
# full_split_df['action'] = full_split_df['action-extraction']
# full_split_df['tissue'] = full_split_df['tissue-extraction']

# full_split_df['instrument'] = full_split_df['instrument_pred']
# full_split_df['action'] = full_split_df['action_pred']
# full_split_df['tissue'] = full_split_df['tissue_pred']

full_split_df = full_split_df.fillna('NONE')
full_split_df = full_split_df[(full_split_df['instrument'] != 'UNCERTAIN') |
                              (full_split_df['action'] != 'UNCERTAIN') |
                              (full_split_df['tissue'] != 'UNCERTAIN')]
full_split_df = full_split_df[(full_split_df['instrument'] != 'NONE') |
                              (full_split_df['action'] != 'NONE') |
                              (full_split_df['tissue'] != 'NONE')]
print(full_split_df.shape)

# print(f"Running iat=extractions, yes frames")
# print(f"Running iat=extractions, no frames")
# print(f"Running iat=clusters, yes frames")
# print(f"Running iat=clusters, no frames, task-specific examples")

print("Running iat=pred, no frames")
# print("Running iat=pred, yes frames")

# print("Running no_iat, yes frames")

# chatllm_outputs_dir = '~/surgery/surgical_fb_generation/SurgFBGen/ChatLLM_outputs/gpt-4o/20250831_173231'
# paths = sorted([os.path.join(chatllm_outputs_dir, p) for p in os.listdir(chatllm_outputs_dir) if p.endswith('.json')])
# feedback_outputs = []
# for path in paths:
#     with open(path, 'r') as f:
#         feedback_outputs.append(json.load(f)['output']['response_text'])
# feedback_outputs += [None] * (len(full_split_df) - len(feedback_outputs))
# full_split_df['feedback'] = feedback_outputs
# full_split_df = full_split_df.reset_index(drop=True)

config = FeedbackGeneratorConfig(
    input_frames=False,
    input_iat_triplet=True,
    input_class_definitions=True,
    clips_dir=clips_dir,
    num_reference_examples=10,
    reference_examples_granularity='all',
    # reference_examples_granularity='task',
    chatllm_name='gpt-4o',
    temperature=0.2,
    max_tokens=10000,
)

feedback_generator = FeedbackGenerator(
    config=config,
    api_key=os.environ['OPENAI_API_KEY'],  # or 'GOOGLE_API_KEY' for Gemini
    instrument_col='instrument_pred',
    action_col='action_pred',
    tissue_col='tissue_pred',
)

feedback_df = feedback_generator.generate_all_feedback(
    full_split_df,
)


# feedback_df.to_csv('~/surgery/surgical_fb_generation/SurgFBGen/data/iat_predictor_splits/feedback_df-iat=extractions-yes_frames.csv', index=False)
# feedback_df.to_csv('~/surgery/surgical_fb_generation/SurgFBGen/data/iat_predictor_splits/feedback_df-iat=extractions-no_frames.csv', index=False)

# feedback_df.to_csv('~/surgery/surgical_fb_generation/SurgFBGen/data/iat_predictor_splits/feedback_df-iat=cluster-yes_frames.csv', index=False)
# feedback_df.to_csv('~/surgery/surgical_fb_generation/SurgFBGen/data/iat_predictor_splits/feedback_df-iat=cluster-no_frames.csv', index=False)

# feedback_df.to_csv('~/surgery/surgical_fb_generation/SurgFBGen/data/iat_predictor_splits/feedback_df-no_iat-yes_frames.csv', index=False)

# feedback_df.to_csv('~/surgery/surgical_fb_generation/SurgFBGen/data/iat_predictor_splits/feedback_df-iat=cluster-no_frames-task_examples.csv', index=False)



# feedback_df.to_csv('~/surgery/surgical_fb_generation/SurgFBGen/outputs/feedback_generations/feedback_df-iat=pred-no_frames.csv', index=False)
# feedback_df.to_csv('~/surgery/surgical_fb_generation/SurgFBGen/outputs/feedback_generations/feedback_df-iat=pred-yes_frames.csv', index=False)

# feedback_df.to_csv('~/surgery/surgical_fb_generation/SurgFBGen/outputs/feedback_generations/feedback_df-iat=pred-no_frames-final.csv', index=False)
feedback_df.to_csv('~/surgery/surgical_fb_generation/SurgFBGen/outputs/feedback_generations/feedback_df-iat=pred-no_frames-final-modified_prompt.csv', index=False)
# feedback_df.to_csv('~/surgery/surgical_fb_generation/SurgFBGen/outputs/feedback_generations/feedback_df-iat=pred-no_frames-no_tracks.csv', index=False)
