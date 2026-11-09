# Data Directory

This folder contains dataset files for the Amazon Review Sentiment Analysis project.

## Included Files
- `sample_reviews.csv`: A lightweight benchmark dataset of 20 realistic Amazon food reviews across ratings 1 to 5 with full metadata (`Id`, `ProductId`, `UserId`, `ProfileName`, `HelpfulnessNumerator`, `HelpfulnessDenominator`, `Score`, `Summary`, `Text`). This allows the Streamlit app and pipeline to run immediately upon cloning without requiring a large external download.

## Full Dataset
The project is built on the **Amazon Fine Food Reviews** dataset:
- Source: [Kaggle - Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- Total Reviews: ~568,454
- Features:
  - `Id`: Row index
  - `ProductId`: Unique Amazon ASIN identifier
  - `UserId`: Unique reviewer identifier
  - `ProfileName`: Reviewer profile name
  - `HelpfulnessNumerator`: Number of users who found the review helpful
  - `HelpfulnessDenominator`: Total number of users who voted on helpfulness
  - `Score`: Rating between 1 and 5
  - `Time`: Timestamp
  - `Summary`: Brief review title
  - `Text`: Full text of the customer review

To train on the full dataset:
1. Download `Reviews.csv` from Kaggle.
2. Place `Reviews.csv` in this directory (`data/Reviews.csv`).
3. Run `python train.py --data data/Reviews.csv`.
