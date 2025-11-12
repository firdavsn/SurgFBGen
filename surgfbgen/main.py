"""
SurgFBGen Main Pipeline Script
==============================

This script provides end-to-end examples for running the complete surgical feedback
generation pipeline. It demonstrates how to use the SurgFBGen library to:

1. Extract multi-modal embeddings (vision, text, motion) from surgical data
2. Train IAT (Instrument-Action-Tissue) predictors
3. Generate predictions on surgical video clips
4. Generate natural language feedback based on IAT predictions
5. Evaluate the quality of generated feedback

Usage:
    To run the complete pipeline:
        $ python main.py
    
    To run specific stages, uncomment the desired function calls in main().
    Each example function is self-contained and can be run independently
    if the required dependencies (embeddings, models, etc.) exist.

Required Environment Variables:
    REPO_DIR: Path to the SurgFBGen repository root
    CLIPS_DATA_DIR: Directory containing surgical video clips
    DATA_DIR: Directory containing annotation and reference data
    CHECKPOINTS_DIR: Directory containing model checkpoints
    OPENAI_API_KEY: API key for OpenAI services

Configuration:
    All paths are configured using environment variables. Set these in .env before
    running the script to work with your own data.

Requirements:
    - CUDA-capable GPU 
    - Pre-trained model checkpoints (CoTracker, SurgVLP, etc.)
    - OpenAI API key set in environment variable OPENAI_API_KEY
    - Input data organized according to expected structure
"""

import os
from typing import Dict

import torch

# Import all necessary components from surgfbgen
from surgfbgen.interface import (
    # Enumerations
    VisionModelName,
    TextModelName,
    TextType,
    IATColumn,
    SurgicalVLModel,
    IATInputs,
    
    # Feature extraction functions
    extract_vis_embs,
    extract_text_embs,
    extract_motion_embeddings,
    
    # IAT prediction functions
    run_iat_predictor_train_and_eval,
    train_via_hybrid,
    run_predictions,
    run_all_predictions_and_save,
    
    # Feedback generation and evaluation
    generate_feedback,
    evaluate_feedback,
    
    # Model classes
    IATPredictor,
)


def example_extract_vision_embeddings():
    """
    Example: Extract visual embeddings from surgical video clips.
    
    This function demonstrates how to extract frame-level visual features
    using the SurgVLP model. The embeddings are saved in HDF5 format.
    """
    extract_vis_embs(
        vision_model=VisionModelName.SURGVLP,
        clips_data_dir=os.environ['CLIPS_DATA_DIR'],
        output_embeddings_dir=os.path.join(
            os.environ['REPO_DIR'],
            'outputs/example/embeddings'
        ),
    )
    

def example_extract_text_embeddings():
    """
    Example: Extract text embeddings from procedure and task descriptions.
    
    This function shows how to generate semantic embeddings for both
    surgical procedures and specific tasks using the SurgVLP text encoder.
    """
    # Extract procedure embeddings
    extract_text_embs(
        text_model=TextModelName.SURGVLP,
        input_csv_path=os.path.join(
            os.environ['REPO_DIR'],
            'data/iat/procedures_df.csv'
        ),
        output_embeddings_dir=os.path.join(
            os.environ['REPO_DIR'],
            'outputs/example/embeddings'
        ),
        text_type=TextType.PROCEDURE,
    )
    
    # Extract task embeddings
    extract_text_embs(
        text_model=TextModelName.SURGVLP,
        input_csv_path=os.path.join(
            os.environ['REPO_DIR'],
            'data/iat/tasks_df.csv'
        ),
        output_embeddings_dir=os.path.join(
            os.environ['REPO_DIR'],
            'outputs/example/embeddings'
        ),
        text_type=TextType.TASK,
    )


def example_extract_motion_embeddings():
    """
    Example: Extract motion embeddings using CoTracker.
    
    This function demonstrates instrument tracking across video frames
    to capture motion patterns for improved IAT prediction.
    """
    extract_motion_embeddings(
        clips_dir=os.environ['CLIPS_DATA_DIR'],
        output_h5_path=os.path.join(
            os.environ['REPO_DIR'],
            'outputs/example/embeddings/motion/instrument_tracks-num_tracks=15.h5'
        ),
        cotracker_checkpoint_path=os.path.join(
            os.environ['CHECKPOINTS_DIR'],
            'cotracker3.pth'
        ),
        overwrite=True,
        filter_instrument=True,
        filter_instrument_num_tracks=15
    )


def example_train_and_eval_iat_predictor():
    """
    Example: Train and evaluate a single IAT predictor with cross-validation.
    
    This function shows how to train an instrument predictor using
    k-fold cross-validation and save evaluation metrics.
    """
    run_iat_predictor_train_and_eval(
        iat_col=IATColumn.INSTRUMENT,
        model=SurgicalVLModel.SURGVLP,
        inputs=IATInputs.VISION_PROCEDURE_TASK,
        output_json_path=os.path.join(
            os.environ['REPO_DIR'],
            'outputs/example/iat_metrics/instrument-surgvlp-vision+procedure+task-none=100.json'
        ),
        annotations_path=os.path.join(
            os.environ['REPO_DIR'],
            'data/iat_predictor_splits/full.csv'
        ),
        embeddings_dir=os.path.join(
            os.environ['REPO_DIR'],
            'outputs/example/embeddings'
        ),
        pred_csv_path=os.path.join(
            os.environ['REPO_DIR'],
            'outputs/example/iat_predictions/instrument-surgvlp-vision+procedure+task-none=100.csv'
        ),
        num_none_included=100,
        seed=0,
        num_tracks=15,
        metric_avg='macro',
        multiple_instance_training=False,
    )


def example_train_iat_predictors():
    """
    Example: Train all three IAT predictors (instrument, action, tissue).
    
    This function demonstrates training separate models for each IAT component
    using multi-modal inputs (vision + procedure + task + motion).
    
    Returns:
        Tuple of (predictors dict, surgical_model enum)
    """
    print("=" * 60)
    print("Training IAT Predictors")
    print("=" * 60)
    
    # Configuration for training
    surgical_model = SurgicalVLModel.SURGVLP  
    inputs = IATInputs.VISION_PROCEDURE_TASK
    embeddings_dir = os.path.join(
        os.environ['REPO_DIR'],
        'outputs/example/embeddings'
    )
    
    predictors = {}
    for iat_col in [IATColumn.INSTRUMENT, IATColumn.ACTION, IATColumn.TISSUE]:
        print(f"\n{'-' * 40}\nTraining for IAT Column: {iat_col.value}\n{'-' * 40}")
    
        # Paths for data and outputs
        model_save_path = os.path.join(
            os.environ['REPO_DIR'],
            f'outputs/example/models/iat_predictor_{iat_col.value}_{surgical_model.value.lower()}_{inputs.value.replace("+", "_")}_tracks.pth'
        )
        annotations_path = os.path.join(
            os.environ['REPO_DIR'],
            'data/iat_predictor_splits/full.csv'
        )
        vision_embeddings_dir = os.path.join(embeddings_dir, 'vision')
        procedures_embs_path = os.path.join(
            embeddings_dir, 
            'text', 
            f'{surgical_model.value.lower()}_procedure_embs.parquet'
        )
        tasks_embs_path = os.path.join(
            embeddings_dir, 
            'text', 
            f'{surgical_model.value.lower()}_task_embs.parquet'
        )
        instrument_tracks_path = os.path.join(
            embeddings_dir, 
            'motion', 
            'instrument_tracks-num_tracks=15.h5'
        )
        
        # Training parameters
        training_params = {
            'iat_col': iat_col.value,
            'model': surgical_model.value.lower(),
            'inputs': inputs.value,
            'model_save_path': model_save_path,
            'num_none_included': 100,
            'train_test_split_ratio': 0.8,
            'use_validation_set': True,
            'vision_embeddings_dir': vision_embeddings_dir,
            'annotations_path': annotations_path,
            'procedures_embs_path': procedures_embs_path,
            'tasks_embs_path': tasks_embs_path,
            'instrument_tracks_path': instrument_tracks_path,
            'seed': 42,
            'num_tracks': 15,
            'multiple_instance_training': False,
            'decouple_mlp': False,
            'metric_avg': 'macro',
            'epochs': 20,
            'batch_size': 8,
            'learning_rate': 0.001,
            'lstm_hidden_dim': 32,
            'num_lstm_layers': 10,
            'dropout': 0.2,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        }
        
        # Train the model
        predictor = train_via_hybrid(**training_params)
        predictors[iat_col.value] = predictor
    
    return predictors, surgical_model


def example_predict_iat(
    predictors: Dict[str, IATPredictor],
    surgical_model: SurgicalVLModel,
):
    """
    Example: Generate IAT predictions on test data and save combined results.
    
    This function shows how to:
    1. Run predictions on a small test set for verification
    2. Generate predictions for all samples
    3. Combine predictions from all three IAT models into one CSV
    
    Args:
        predictors: Dictionary mapping IAT column names to trained predictors
        surgical_model: The surgical VL model used for training
    """
    print("\n" + "=" * 60)
    print("Running Predictions on Test Samples")
    print("=" * 60)
    
    # Run predictions on a small test set first
    for iat_col, predictor in predictors.items():
        print(f"\n{'-' * 40}\nRunning TEST predictions for IAT Column: {iat_col}\n{'-' * 40}")
        
        if hasattr(predictor, 'processed_df'):
            test_df = predictor.processed_df.tail(10)
            
            test_results = run_predictions(
                predictor=predictor,
                data_df=test_df,
                verbose=True
            )
            
            test_output_path = os.path.join(
                os.environ['REPO_DIR'],
                f'outputs/example/predictions/test_predictions_{iat_col}_{surgical_model.value.lower()}.csv'
            )
            os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
            test_results.to_csv(test_output_path, index=False)
            print(f"\nSaved test predictions to: {test_output_path}")
    
    # Save all predictions combined
    print("\n" + "=" * 60)
    print("Saving All COMBINED Predictions to CSV")
    print("=" * 60)
    
    full_output_path = os.path.join(
        os.environ['REPO_DIR'],
        f'outputs/example/predictions/full_predictions_combined_{surgical_model.value.lower()}.csv'
    )
    
    full_results = run_all_predictions_and_save(
        predictors=predictors,
        output_path=full_output_path,
        verbose=True,
        metric_avg='macro'
    )
    
    return full_results


def example_generate_feedback(
    surgical_model: SurgicalVLModel = SurgicalVLModel.SURGVLP,
):
    """
    Example: Generate natural language feedback from IAT predictions.
    
    This function demonstrates using GPT-4 to generate surgical feedback
    based on predicted IAT triplets, optionally including reference examples
    and class definitions.
    
    Args:
        surgical_model: The surgical VL model used for predictions
    """
    print("\n" + "=" * 60)
    print("Example: Generating Feedback")
    print("=" * 60)

    clips_dir = os.environ['CLIPS_DATA_DIR']
    predictions_path = os.path.join(
        os.environ['REPO_DIR'],
        f'outputs/example/predictions/full_predictions_combined_{surgical_model.value.lower()}.csv'
    )
    output_path = os.path.join(
        os.environ['REPO_DIR'],
        'outputs/example/feedback_generations/feedback_df-iat=pred-no_frames-final-modified_prompt.csv'
    )

    # FeedbackGenerator configuration
    config_params = {
        'input_frames': False,
        'input_iat_triplet': True,
        'input_class_definitions': True,
        'num_reference_examples': 10,
        'reference_examples_granularity': 'all',
        'chatllm_name': 'gpt-4.1-mini',
        'temperature': 0.2,
        'max_tokens': 10000,
    }
    
    # IAT prediction column names
    iat_cols = {
        'iat_instrument_col': 'instrument_pred',
        'iat_action_col': 'action_pred',
        'iat_tissue_col': 'tissue_pred',
    }

    generate_feedback(
        predictions_csv_path=predictions_path,
        output_csv_path=output_path,
        clips_dir=clips_dir,
        config_params=config_params,
        api_key_env_var='OPENAI_API_KEY',
        **iat_cols
    )


def example_evaluate_feedback():
    """
    Example: Evaluate generated feedback quality against ground truth.
    
    This function shows how to use an LLM-based evaluator to score
    the generated feedback by comparing it with ground truth annotations.
    """
    print("\n" + "=" * 60)
    print("Example: Evaluating Feedback")
    print("=" * 60)

    feedback_path = os.path.join(
        os.environ['REPO_DIR'],
        'outputs/example/feedback_generations/feedback_df-iat=pred-no_frames-final-modified_prompt.csv'
    )
    scores_path = os.path.join(
        os.environ['REPO_DIR'],
        'outputs/example/feedback_evaluations/scores_df-iat=pred-no_frames-final-modified_prompt.csv'
    )

    # FeedbackEvaluator configuration
    config_params = {
        'chatllm_name': 'gpt-4.1-mini',
        'temperature': 0.2,
        'max_tokens': 10000,
    }
    
    # Column specifications
    column_names = {
        'ground_truth_col': 'dialogue',
        'generated_col': 'feedback',
    }

    evaluate_feedback(
        feedback_csv_path=feedback_path,
        output_csv_path=scores_path,
        config_params=config_params,
        api_key_env_var='OPENAI_API_KEY',
        save_pickle=True,
        **column_names
    )


def main():
    """
    Main pipeline execution function.
    
    Uncomment the desired stages to run. The pipeline stages are:
    1. Feature extraction (vision, text, motion)
    2. IAT predictor training
    3. Prediction generation
    4. Feedback generation
    5. Feedback evaluation
    """
    
    # ================================
    # Stage 1: Extract Features
    # ================================
    # example_extract_vision_embeddings()
    # example_extract_text_embeddings()   
    # example_extract_motion_embeddings()
        
    # ================================
    # Stage 2: Train and Evaluate
    # ================================
    # example_train_and_eval_iat_predictor()  # Single predictor with cross-validation
    
    # ================================
    # Stage 3: Train All Predictors
    # ================================
    # predictors, surgical_model = example_train_iat_predictors()
    
    # ================================
    # Stage 4: Generate Predictions
    # ================================
    # example_predict_iat(predictors, surgical_model)
    
    # ================================
    # Stage 5: Generate Feedback
    # ================================
    example_generate_feedback()
    
    # ================================
    # Stage 6: Evaluate Feedback
    # ================================
    example_evaluate_feedback()
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()