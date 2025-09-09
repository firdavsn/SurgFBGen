# SurgFBGen

An automated natural language surgical feedback generation system for robotic urological procedures. The system analyzes surgical video frames to predict instrument-action-tissue (IAT) triplets to ground an LLM to generate concise, actionable feedback for surgical trainees. It supports multiple input modalities (video frames, + procedure and task context, + instrument motion tracking) and provides structured feedback evaluation through an automated scoring mechanism.

## Project Structure

```
SurgFBGen/
├── surgfbgen/                    # Main package
│   ├── config/                   # Configuration files
│   │   ├── environment.py        # Environment settings
│   │   └── surgical_VL_models/   # Vision-language model configs
│   ├── models/                   # Core models
│   │   ├── feedback_generator.py # Main feedback generation model
│   │   ├── feedback_evaluator.py # Feedback evaluation model
│   │   ├── iat_predictor.py      # Instrument-action-tissue predictor
│   │   └── utils.py              # Model utilities
│   ├── prompts/                  # Prompt templates and interfaces
│   │   ├── chatllm_interface.py  # LLM interface
│   │   ├── templates/            # Prompt templates
│   │   └── cli.py                # Command-line interface
│   ├── scripts/                  # Training and evaluation scripts
│   │   ├── generate_feedback.py  # Feedback generation script
│   │   ├── evaluate_feedback.py  # Feedback evaluation script
│   │   └── iat_predictor_train_and_eval.py # IAT predictor training
│   └── notebooks/                # Jupyter notebooks for analysis
├── data/                         # Dataset directories
│   ├── iat/                      # IAT triplet data
│   ├── CholecT45-related/        # CholecT45 dataset
│   └── urology-related/          # Urology dataset
├── outputs/                      # Generated outputs
├── checkpoints/                  # Model checkpoints
├── ChatLLM_outputs/              # LLM-generated outputs
├── constants.json                # Project constants
├── keys.json                     # API keys configuration
└── setup.py                      # Package setup
```