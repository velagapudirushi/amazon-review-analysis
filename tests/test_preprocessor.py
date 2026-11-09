"""
Unit tests for text preprocessor and sentiment pipeline.
"""

import os
import unittest
from src.preprocessor import ReviewPreprocessor, score_to_sentiment
from src.model import ReviewSentimentPipeline
from src.utils import load_or_create_sample_data


class TestReviewPreprocessor(unittest.TestCase):

    def setUp(self):
        self.preprocessor = ReviewPreprocessor()

    def test_html_and_url_removal(self):
        raw = "Check this <b>amazing</b> product at https://example.com/item and www.amazon.com!"
        cleaned = self.preprocessor.clean_text(raw)
        self.assertNotIn("<b>", cleaned)
        self.assertNotIn("https", cleaned)
        self.assertNotIn("www", cleaned)
        self.assertIn("amaze", cleaned)

    def test_negation_preservation(self):
        raw = "This was not good and never arrived. No one liked it."
        cleaned = self.preprocessor.clean_text(raw)
        # Ensure 'not', 'never', 'no' are preserved
        self.assertIn("not", cleaned)
        self.assertIn("never", cleaned)
        self.assertIn("no", cleaned)

    def test_standard_stopwords_removal(self):
        raw = "This is a product for the family in the kitchen."
        analysis = self.preprocessor.analyze_tokens(raw)
        # 'is', 'a', 'for', 'the', 'in' should be removed
        self.assertIn("the", analysis["removed_stopwords"])

    def test_score_to_sentiment(self):
        self.assertEqual(score_to_sentiment(5), "Positive")
        self.assertEqual(score_to_sentiment(4), "Positive")
        self.assertEqual(score_to_sentiment(3), "Neutral")
        self.assertEqual(score_to_sentiment(2), "Negative")
        self.assertEqual(score_to_sentiment(1), "Negative")


class TestPipeline(unittest.TestCase):

    def test_pipeline_train_and_predict(self):
        df = load_or_create_sample_data()
        pipeline = ReviewSentimentPipeline(max_features=500, ngram_range=(1, 2))
        metrics = pipeline.train(df, text_col="Text", target_col="Score", test_size=0.2)

        self.assertIn("accuracy", metrics)
        self.assertTrue(pipeline.is_trained)

        res = pipeline.predict_single("Delicious coffee, absolutely loved it!")
        self.assertIn("predicted_score", res)
        self.assertIn("sentiment", res)
        self.assertIn(res["sentiment"], ["Positive", "Neutral", "Negative"])


if __name__ == "__main__":
    unittest.main()
