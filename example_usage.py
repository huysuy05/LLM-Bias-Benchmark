#!/usr/bin/env python3
"""
Example usage of the eval_with_hgf_few_shots.py script.

This script demonstrates how to use the LLM evaluator programmatically.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import the evaluator
sys.path.append(str(Path(__file__).parent / "src"))

from eval_with_hgf_few_shots import LLMEvaluator

def main():
    """Example usage of the LLMEvaluator class."""
    
    # Initialize the evaluator with a specific model
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    evaluator = LLMEvaluator(model_name, device="cuda")  # or "mps" for Apple Silicon
    
    # Authenticate with HuggingFace (optional, but recommended)
    # You can set HF_TOKEN environment variable or pass token directly
    evaluator.authenticate_hf()
    
    # Example 1: Load and evaluate AG News dataset
    print("Loading AG News datasets...")
    ag_news_datasets = evaluator.load_ag_news_data("Data/ag_news")
    
    # Run experiments on AG News with specific shot counts
    print("Running experiments on AG News...")
    results = evaluator.run_experiments(
        ag_news_datasets, 
        'ag_news', 
        evaluator.label_maps['ag_news'],
        shots_list=[0, 2, 4],  # Fewer shots for faster execution
        batch_size=8  # Smaller batch size for memory efficiency
    )
    
    print("AG News evaluation completed!")
    print(f"Results shape: {results.shape}")
    print("\nSample results:")
    print(results[['dataset', 'shots', 'macro_f1', 'balanced_accuracy']].head())
    
    # Example 2: Load and evaluate Toxic Text dataset
    print("\n" + "="*50)
    print("Loading Toxic Text datasets...")
    toxic_datasets = evaluator.load_toxic_text_data("Data/toxic_text")
    
    print("Running experiments on Toxic Text...")
    toxic_results = evaluator.run_experiments(
        toxic_datasets,
        'toxic_text',
        evaluator.label_maps['toxic_text'],
        shots_list=[0, 2],
        batch_size=8
    )
    
    print("Toxic Text evaluation completed!")
    print(f"Results shape: {toxic_results.shape}")
    
    # Example 3: Load and evaluate Twitter Emotion dataset
    print("\n" + "="*50)
    print("Loading Twitter Emotion datasets...")
    emotion_datasets = evaluator.load_twitter_emotion_data("Data/twit")
    
    print("Running experiments on Twitter Emotion...")
    emotion_results = evaluator.run_experiments(
        emotion_datasets,
        'twitter_emotion',
        evaluator.label_maps['twitter_emotion'],
        shots_list=[0, 2],
        batch_size=8
    )
    
    print("Twitter Emotion evaluation completed!")
    print(f"Results shape: {emotion_results.shape}")
    
    print("\nAll evaluations completed successfully!")
    print("Check the 'results/' directory for detailed CSV outputs.")

if __name__ == "__main__":
    main()

