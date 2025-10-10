import os
import pandas as pd
from surgfbgen.models.iat_predictor import IATGPTPredictor, IATPredictorGPTConfig

def main():
    """
    Main function to run the IAT prediction process.
    """
    # 1. Define constants and paths
    # Use os.path.expanduser to correctly handle the '~' symbol
    clips_dir = os.path.expanduser('~/surgery/clips_with_wiggle/fb_clips_wiggle')
    
    # This should be the path to your input data that has 'cvid', 'procedure', etc.
    input_data_path = os.path.expanduser('~/surgery/surgical_fb_generation/SurgFBGen/data/iat_predictor_splits/full.csv')
    
    # Define where the output CSV with predictions will be saved
    output_dir = os.path.expanduser('~/surgery/surgical_fb_generation/SurgFBGen/outputs/iat_predictions/')
    os.makedirs(output_dir, exist_ok=True)
    output_filename = 'iat_predictions-gpt-4o.csv'
    output_data_path = os.path.join(output_dir, output_filename)

    print(f"Loading data from: {input_data_path}")
    data_df = pd.read_csv(input_data_path)
    print(f"Loaded {len(data_df)} rows.")
    
    # For demonstration, you might want to run on a smaller subset first
    # data_df = data_df.head(5)

    # 2. Configure the IAT Predictor
    print("Configuring IAT Predictor...")
    config = IATPredictorGPTConfig(
        input_frames=True,          # Set to True to use video frames, False otherwise
        input_procedure=True,
        input_task=True,
        clips_dir=clips_dir,
        chatllm_name='gpt-4o',       # Or any other compatible model like 'gemini-1.5-pro'
        temperature=0.2,
        max_tokens=500,             # Max tokens for the JSON output should be relatively small
    )

    # 3. Initialize the IAT Predictor
    # Ensure you have your API key set as an environment variable
    try:
        api_key = os.environ['OPENAI_API_KEY']
    except KeyError:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    print(f"Initializing predictor with model: {config.chatllm_name}")
    predictor = IATGPTPredictor(
        config=config,
        api_key=api_key,
    )

    # 4. Generate IAT triplets for all rows in the DataFrame
    print("Generating IAT triplet predictions...")
    predictions_df = predictor.generate_all_iat_triplets(
        data_df=data_df,
        override_existing=True # Set to True to re-generate even if 'iat_triplet' column exists
    )

    # 5. Save the results to a new CSV file
    print(f"Saving predictions to: {output_data_path}")
    predictions_df.to_csv(output_data_path, index=False)
    print("Prediction process complete.")

if __name__ == '__main__':
    main()