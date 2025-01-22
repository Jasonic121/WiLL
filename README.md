# WiLL System

A comprehensive research system for conducting experiments with Large Language Models (LLMs), focusing on ablation studies and model evaluation.

## Project Structure

- `AblationStudyExp/`: Contains various experimental setups for ablation studies
  - `Baseline/`: Baseline experiments with different models (GPT, LLaMA)
  - `Chunking/`: Experiments with different text chunking strategies
  - `SimilarityFunction/`: Studies on various similarity measurement approaches
- `csv_files/`: Data files for experiments
- `Finetune/`: Model fine-tuning scripts and configurations
- `environment/`: Environment setup and configuration files
- `outputEvaluation/`: Experimental results and evaluation metrics

## Setup

1. Clone the repository
2. Set up the environment:
   ```bash
   # Create and activate a virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```
3. Configure environment variables in `.env` file

## Running Experiments

The system supports various experimental configurations:
- Baseline model comparisons
- Chunking strategy evaluation
- Similarity function analysis

Each experiment can be run through the respective Jupyter notebooks in the `AblationStudyExp` directory.

## Features

- Multiple model support (GPT, LLaMA)
- Automated experiment notification system
- Comprehensive evaluation metrics
- Flexible experimental pipeline
- Cross-platform compatibility

## Requirements

- Python 3.8+
- Jupyter Notebook/Lab
- Required packages are listed in `requirements.txt`