# Models Directory

This directory stores serialized model artifacts.

## Files
- `sentiment_pipeline.joblib`: Serialized dictionary containing:
  - `vectorizer`: Trained `TfidfVectorizer` (ngram_range=(1, 2), sublinear_tf=True)
  - `classifier`: Trained `LogisticRegression` with `class_weight='balanced'`
  - `classes_`: Rating class labels `[1, 2, 3, 4, 5]`
  - `is_trained`: Boolean status

To retrain and overwrite the model, use:
```bash
python train.py --data data/sample_reviews.csv --output models/sentiment_pipeline.joblib
```
