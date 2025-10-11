import os
import json
import pandas as pd

from surgfbgen.models.feedback_evaluator import FeedbackEvaluator, FeedbackEvaluatorConfig

clips_dir = '~/surgery/clips_with_wiggle/fb_clips_wiggle'

# feedback_path = '~/surgery/surgical_fb_generation/SurgFBGen/data/iat_predictor_splits/feedback_df-iat=extractions-yes_frames.csv'
# feedback_path = '~/surgery/surgical_fb_generation/SurgFBGen/data/iat_predictor_splits/feedback_df-iat=extractions-no_frames.csv'

# feedback_path = '~/surgery/surgical_fb_generation/SurgFBGen/data/iat_predictor_splits/feedback_df-iat=cluster-yes_frames.csv'
# feedback_path = '~/surgery/surgical_fb_generation/SurgFBGen/data/iat_predictor_splits/feedback_df-iat=cluster-no_frames.csv'

# feedback_path = '~/surgery/surgical_fb_generation/SurgFBGen/outputs/feedback_generations/feedback_df-no_iat-yes_frames.csv'

# feedback_path = '~/surgery/surgical_fb_generation/SurgFBGen/surgfbgen/notebooks/outputs/llama-3.2-11b---10frames_proc+task/inference_results.csv'
# feedback_path = '~/surgery/surgical_fb_generation/SurgFBGen/surgfbgen/notebooks/outputs/llama-3.2-11b---10frames_proc+task_strict_instruction/inference_results.csv'

# feedback_path = '~/surgery/surgical_fb_generation/SurgFBGen/outputs/feedback_generations/feedback_df-iat=pred-no_frames-no_calibration-num_tracks=15.csv'
# feedback_path = '~/surgery/surgical_fb_generation/SurgFBGen/outputs/feedback_generations/feedback_df-iat=pred-no_frames.csv'
# feedback_path = '~/surgery/surgical_fb_generation/SurgFBGen/outputs/feedback_generations/feedback_df-iat=pred-yes_frames.csv'

# feedback_path = '~/surgery/surgical_fb_generation/SurgFBGen/outputs/feedback_generations/feedback_df-iat=pred-no_frames-final.csv'
# feedback_path = '~/surgery/surgical_fb_generation/SurgFBGen/outputs/feedback_generations/feedback_df-iat=pred-no_frames-final-modified_prompt.csv'
# feedback_path = '~/surgery/surgical_fb_generation/SurgFBGen/outputs/feedback_generations/feedback_df-iat=pred-no_frames-no_tracks.csv'

feedback_path = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/outputs/feedback_generations/qwen_vl-frame_wise_voting.csv'

feedback_df = pd.read_csv(feedback_path)

# print(f"Running iat=extractions, yes frames")
# print(f"Running iat=extractions, no frames")
# print(f"Running iat=clusters, yes frames")
# print(f"Running iat=clusters, no frames")

# print("Running no_iat, yes frames")

print("Running iat=pred, no frames")
# print("Running iat=pred, yes frames")

# print("Running Llama-3.2-11b base")

config = FeedbackEvaluatorConfig(
    chatllm_name='gpt-4.1-mini',
    # chatllm_name='gpt-4o',
    # chatllm_name='gemini-2.5-flash',
    temperature=0.2,
    max_tokens=10000,
)

feedback_evaluator = FeedbackEvaluator(
    config=config,
    api_key=os.environ['OPENAI_API_KEY'],  # or 'GOOGLE_API_KEY' for Gemini
    # api_key=os.environ['GOOGLE_API_KEY'],  # or 'GOOGLE_API_KEY' for Gemini
)

scores_df = feedback_evaluator.generate_all_scores(
    feedback_df,
    # gt_fb_col='ground_truth',
    gt_fb_col='dialogue',
    # gen_fb_col='prediction'
    gen_fb_col='feedback'
)


# scores_df.to_csv('~/surgery/surgical_fb_generation/SurgFBGen/results/feedback_evaluations/scores_df-iat=extractions-yes_frames.csv', index=False)
# scores_df.to_csv('~/surgery/surgical_fb_generation/SurgFBGen/results/feedback_evaluations/scores_df-iat=extractions-no_frames.csv', index=False)

# scores_df.to_csv('~/surgery/surgical_fb_generation/SurgFBGen/results/feedback_evaluations/scores_df-iat=cluster-yes_frames.csv', index=False)
# scores_df.to_csv('~/surgery/surgical_fb_generation/SurgFBGen/results/feedback_evaluations/scores_df-iat=cluster-no_frames.csv', index=False)

# scores_path = '~/surgery/surgical_fb_generation/SurgFBGen/results/feedback_evaluations/scores_df-no_iat-yes_frames.csv'
# scores_path = '~/surgery/surgical_fb_generation/SurgFBGen/results/feedback_evaluations/scores_df-no_iat-yes_frames-gemini_25_flash.csv'
scores_path = '/home/firdavs/surgery/surgical_fb_generation/SurgFBGen/results/feedback_evaluations/scores_df-qwen_vl-frame_wise_voting.csv'

# scores_path = feedback_path.replace('inference_results.csv', 'scores_df.csv')

# scores_path = '~/surgery/surgical_fb_generation/SurgFBGen/outputs/feedback_evaluations/scores_df-iat=pred-no_frames-no_calibration-num_tracks=15.csv'
# scores_path = '~/surgery/surgical_fb_generation/SurgFBGen/outputs/feedback_evaluations/scores_df-iat=pred-no_frames.csv'
# scores_path = '~/surgery/surgical_fb_generation/SurgFBGen/outputs/feedback_evaluations/scores_df-iat=pred-yes_frames.csv'
# scores_path = '~/surgery/surgical_fb_generation/SurgFBGen/outputs/feedback_evaluations/scores_df-iat=pred-yes_frames-gemini_25_flash.csv'

# scores_path = '~/surgery/surgical_fb_generation/SurgFBGen/outputs/feedback_evaluations/scores_df-iat=pred-no_frames-final-modified_prompt.csv'
# scores_path = '~/surgery/surgical_fb_generation/SurgFBGen/outputs/feedback_evaluations/scores_df-iat=pred-no_frames-final-gemini_25_flash.csv'
# scores_path = '~/surgery/surgical_fb_generation/SurgFBGen/outputs/feedback_evaluations/scores_df-iat=pred-no_frames-no_tracks.csv'

import pickle
with open(scores_path.replace('csv', 'pkl'), 'wb') as f:
    pickle.dump(scores_df, f)
scores_df.to_csv(scores_path, index=False)
