"""
Training and Evaluation CLI Script for Amazon Review Sentiment Analysis.

Usage:
    python train.py --data data/sample_reviews.csv --output models/sentiment_pipeline.joblib
"""

import argparse
import os
import sys
import pandas as pd

from src.model import ReviewSentimentPipeline
from src.utils import load_or_create_sample_data


def main():
    parser = argparse.ArgumentParser(description="Train Amazon Review Sentiment Model")
    parser.add_argument("--data", type=str, default="data/sample_reviews.csv", help="Path to CSV dataset")
    parser.add_argument("--text-col", type=str, default="Text", help="Name of review text column")
    parser.add_argument("--score-col", type=str, default="Score", help="Name of star rating column (1-5)")
    parser.add_argument("--max-features", type=int, default=50000, help="Maximum TF-IDF features")
    parser.add_argument("--output", type=str, default="models/sentiment_pipeline.joblib", help="Output model path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set split ratio")
    args = parser.parse_args()

    print(f"Loading data from: {args.data}")
    if os.path.exists(args.data):
        df = pd.read_csv(args.data)
    else:
        print(f"Dataset {args.data} not found, generating sample reviews dataset...")
        df = load_or_create_sample_data(args.data)

    print(f"Dataset shape: {df.shape}")
    print(f"Rating value counts:\n{df[args.score_col].value_counts().sort_index()}")

    print("Initializing ReviewSentimentPipeline...")
    pipeline = ReviewSentimentPipeline(max_features=args.max_features, ngram_range=(1, 2))

    print(f"Training on {args.text_col} predicting {args.score_col}...")
    metrics = pipeline.train(df, text_col=args.text_col, target_col=args.score_col, test_size=args.test_size)

    print("\n--- Model Evaluation Metrics ---")
    print(f"Accuracy Score:          {metrics['accuracy']:.4f}")
    print(f"Weighted Precision:      {metrics['precision_weighted']:.4f}")
    print(f"Weighted Recall:         {metrics['recall_weighted']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pipeline.save(args.output)
    print(f"\nModel pipeline successfully saved to: {args.output}")


if __name__ == "__main__":
    main()
