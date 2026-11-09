"""
Model Pipeline Module for Amazon Review Sentiment & Rating Prediction.
Combines TF-IDF n-gram vectorization with class-balanced Logistic Regression.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.preprocessor import ReviewPreprocessor, score_to_sentiment


class ReviewSentimentPipeline:
    """
    End-to-end NLP & Machine Learning pipeline for Amazon Review rating
    and sentiment prediction.
    """

    def __init__(
        self,
        max_features: int = 50000,
        ngram_range: Tuple[int, int] = (1, 2),
        random_state: int = 42
    ):
        self.preprocessor = ReviewPreprocessor()
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True
        )
        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            solver="lbfgs"
        )
        self.is_trained: bool = False
        self.classes_: Optional[np.ndarray] = None

    def preprocess_series(self, text_series: pd.Series) -> pd.Series:
        """Cleans a Pandas Series of texts."""
        return text_series.astype(str).apply(self.preprocessor.clean_text)

    def train(
        self,
        df: pd.DataFrame,
        text_col: str = "Text",
        target_col: str = "Score",
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Union[float, dict]]:
        """
        Trains the TF-IDF + Logistic Regression model on a DataFrame.
        """
        df_clean = df.dropna(subset=[text_col, target_col]).copy()
        df_clean["CleanText"] = self.preprocess_series(df_clean[text_col])

        X = df_clean["CleanText"]
        y = df_clean[target_col].astype(int)

        # Check if stratify is possible (requires >=2 samples per class and test_samples >= n_classes)
        value_counts = y.value_counts()
        n_classes = len(value_counts)
        test_samples = int(len(X) * test_size) if isinstance(test_size, float) else test_size
        can_stratify = (value_counts >= 2).all() and n_classes > 1 and test_samples >= n_classes

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if can_stratify else None
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.classifier.fit(X_train_vec, y_train)
        self.is_trained = True
        self.classes_ = self.classifier.classes_

        y_pred = self.classifier.predict(X_test_vec)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classes": [int(c) for c in self.classes_],
            "train_size": len(X_train),
            "test_size": len(X_test)
        }
        return metrics

    def predict_single(self, text: str) -> Dict[str, Union[int, str, float, dict]]:
        """
        Predicts star rating (1-5), sentiment class, and probability distribution
        for a single review string.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained or loaded.")

        clean_text = self.preprocessor.clean_text(text)
        vec = self.vectorizer.transform([clean_text])
        predicted_score = int(self.classifier.predict(vec)[0])
        sentiment = score_to_sentiment(predicted_score)

        probs = {}
        if hasattr(self.classifier, "predict_proba"):
            raw_probs = self.classifier.predict_proba(vec)[0]
            for cls_val, prob in zip(self.classes_, raw_probs):
                probs[f"{int(cls_val)} Star"] = round(float(prob), 4)

        # Extract top influential n-grams present in this text
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        non_zero_indices = vec.indices
        influential_words = []
        if len(non_zero_indices) > 0:
            for idx in non_zero_indices:
                word = feature_names[idx]
                tfidf_val = vec[0, idx]
                influential_words.append({"token": word, "weight": round(float(tfidf_val), 4)})
            influential_words = sorted(influential_words, key=lambda x: x["weight"], reverse=True)[:10]

        return {
            "predicted_score": predicted_score,
            "sentiment": sentiment,
            "cleaned_text": clean_text,
            "probabilities": probs,
            "confidence": max(probs.values()) if probs else 1.0,
            "influential_tokens": influential_words
        }

    def predict_batch(self, texts: List[str]) -> pd.DataFrame:
        """Predicts ratings and sentiment for an iterable of texts."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained or loaded.")

        clean_texts = [self.preprocessor.clean_text(t) for t in texts]
        vecs = self.vectorizer.transform(clean_texts)
        scores = self.classifier.predict(vecs)
        sentiments = [score_to_sentiment(s) for s in scores]

        return pd.DataFrame({
            "Text": texts,
            "CleanText": clean_texts,
            "PredictedScore": scores,
            "PredictedSentiment": sentiments
        })

    def save(self, filepath: str) -> None:
        """Saves model pipeline to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "is_trained": self.is_trained,
            "classes_": self.classes_
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> "ReviewSentimentPipeline":
        """Loads model pipeline from disk."""
        data = joblib.load(filepath)
        instance = cls()
        instance.vectorizer = data["vectorizer"]
        instance.classifier = data["classifier"]
        instance.is_trained = data.get("is_trained", True)
        instance.classes_ = data.get("classes_", instance.classifier.classes_)
        return instance
