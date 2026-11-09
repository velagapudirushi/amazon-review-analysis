"""
Text Preprocessing Module for Amazon Review Sentiment Analysis.
Preserves crucial negation and sentiment-carrying stopwords while cleaning
HTML tags, URLs, and punctuation, followed by multi-POS lemmatization.
"""

import re
from typing import Dict, List, Set, Tuple

# Negation and sentiment modifiers that MUST NOT be removed
PRESERVED_STOPWORDS: List[str] = [
    "not", "no", "nor", "never", "none", "neither",
    "very", "too", "so", "more", "most", "only", "just",
    "but", "against", "without", "until", "if"
]

FALLBACK_ENGLISH_STOPWORDS: Set[str] = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "because", "as", "at", "by", "for", "with",
    "about", "between", "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "other", "some", "such", "own", "same", "than", "will"
}


def get_base_stopwords() -> Set[str]:
    """Safely retrieves stopwords from NLTK with a reliable fallback."""
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            return set(stopwords.words("english"))
        except LookupError:
            try:
                nltk.download("stopwords", quiet=True)
                return set(stopwords.words("english"))
            except Exception:
                return set(FALLBACK_ENGLISH_STOPWORDS)
    except Exception:
        return set(FALLBACK_ENGLISH_STOPWORDS)


class ReviewPreprocessor:
    """Preprocessor for cleaning Amazon review text."""

    def __init__(self, preserved_words: List[str] = None):
        if preserved_words is None:
            preserved_words = PRESERVED_STOPWORDS
        self.preserved_words = set(preserved_words)
        self.stop_words = get_base_stopwords() - self.preserved_words

        # Lemmatizer initialization with fallback
        try:
            import nltk
            from nltk.stem import WordNetLemmatizer
            try:
                self.lemmatizer = WordNetLemmatizer()
                # test lemmatizer call
                self.lemmatizer.lemmatize("testing", pos="v")
            except LookupError:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
                self.lemmatizer = WordNetLemmatizer()
        except Exception:
            self.lemmatizer = None

    def clean_text(self, text: str) -> str:
        """
        Cleans review text:
        1. Lowercases string
        2. Strips HTML markup (<...>)
        3. Removes URLs
        4. Removes punctuation, digits, and special characters
        5. Filters out common stopwords while keeping negation terms
        6. Lemmatizes through verbs, nouns, and adjectives
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        # Lowercase
        text = text.lower()
        # Remove HTML tags
        text = re.sub(r"<.*?>", " ", text)
        # Remove URLs
        text = re.sub(r"https?\S+|www\.\S+", " ", text)
        # Remove non-alphabetic characters
        text = re.sub(r"[^a-z\s]", " ", text)

        tokens = text.split()
        tokens = [w for w in tokens if w not in self.stop_words and len(w) > 1]

        if self.lemmatizer is None:
            return " ".join(tokens)

        # Multi-POS lemmatization: verb -> noun -> adjective
        lemmatized_tokens = []
        for w in tokens:
            try:
                word = self.lemmatizer.lemmatize(w, pos="v")
                word = self.lemmatizer.lemmatize(word, pos="n")
                word = self.lemmatizer.lemmatize(word, pos="a")
                lemmatized_tokens.append(word)
            except Exception:
                lemmatized_tokens.append(w)

        return " ".join(lemmatized_tokens)

    def analyze_tokens(self, text: str) -> Dict[str, object]:
        """
        Returns a diagnostic breakdown of the text showing raw tokens,
        filtered stopwords, preserved negation keywords, and final cleaned text.
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        raw_lower = text.lower()
        cleaned_no_html = re.sub(r"<.*?>", " ", raw_lower)
        cleaned_no_url = re.sub(r"https?\S+|www\.\S+", " ", cleaned_no_html)
        cleaned_alpha = re.sub(r"[^a-z\s]", " ", cleaned_no_url)
        all_words = cleaned_alpha.split()

        preserved_found = [w for w in all_words if w in self.preserved_words]
        removed_stopwords = [w for w in all_words if w in self.stop_words]
        final_cleaned = self.clean_text(text)

        return {
            "original_length": len(text.split()),
            "cleaned_length": len(final_cleaned.split()),
            "preserved_negations": list(set(preserved_found)),
            "removed_stopwords": list(set(removed_stopwords)),
            "cleaned_text": final_cleaned
        }


def score_to_sentiment(score: int) -> str:
    """Maps 1-5 rating score to sentiment label."""
    try:
        score_val = int(score)
    except (ValueError, TypeError):
        return "Unknown"

    if score_val >= 4:
        return "Positive"
    elif score_val == 3:
        return "Neutral"
    else:
        return "Negative"


def sentiment_to_color(sentiment: str) -> str:
    """Returns color hex corresponding to sentiment."""
    mapping = {
        "Positive": "#28a745",
        "Neutral": "#ffc107",
        "Negative": "#dc3545"
    }
    return mapping.get(sentiment, "#6c757d")
