# 📦 Amazon Review Sentiment Analysis & NLP Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-green.svg)](https://www.nltk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Natural Language Processing (NLP) and Machine Learning system for analyzing **568,454 Amazon Fine Food Reviews** across **74,258 products**. The pipeline combines custom negation-preserving text normalization, 3-stage WordNet lemmatization, 50,000 unigram-bigram TF-IDF feature extraction, and class-balanced multi-class Logistic Regression to predict 1–5 star ratings and sentiment classifications, accompanied by an interactive **Streamlit web application**.

---

## 🚀 Key Highlights & Benchmarks

- **Dataset Scale**: **568,454 reviews** across **74,258 products** and **256K+ unique users**.
- **Preprocessing Pipeline**: Strips HTML tags and URLs, removes punctuation/digits, preserves **16 sentiment-critical negation tokens**, and executes **3-stage WordNet lemmatization** (verbs, nouns, adjectives).
- **Severe Class Imbalance Handled**: Discovered a **78.1% positive rating skew** (4–5 stars) in the dataset; solved via cost-sensitive learning (`class_weight='balanced'`).
- **Model Performance**:
  - **77.8%** Weighted Precision
  - **73.2%** Multi-Class Accuracy
  - Sub-millisecond real-time inference latency.
- **Interactive Application**: Full-featured **Streamlit web dashboard** with real-time star prediction, token-level TF-IDF interpretability, batch file processing, and interactive Plotly exploratory data analysis.

---

## 🏛️ System Architecture

```
                                  [ Raw Amazon Review ]
                                            │
                                            ▼
                        ┌───────────────────────────────────────┐
                        │        NLP Preprocessing Engine       │
                        │  • HTML & URL Stripping               │
                        │  • Lowercase & Non-Alpha Removal      │
                        │  • Negation Stopwords Preserved (16)  │
                        │  • 3-Stage WordNet Lemmatization      │
                        └───────────────────┬───────────────────┘
                                            │
                                            ▼
                        ┌───────────────────────────────────────┐
                        │      TF-IDF N-Gram Vectorizer         │
                        │  • 50,000 Unigram & Bigram Features   │
                        │  • Sublinear Term-Frequency Scaling   │
                        └───────────────────┬───────────────────┘
                                            │
                                            ▼
                        ┌───────────────────────────────────────┐
                        │      Balanced Logistic Classifier     │
                        │  • 5-Class Multi-Class Formulation    │
                        │  • Class-Weighted Loss Balancing      │
                        │  • Stratified 80/20 Train-Test Split  │
                        └───────────────────┬───────────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
       [ Star Rating: 1 to 5 ⭐ ]                     [ Sentiment: Pos / Neu / Neg ]
```

---

## 📊 Exploratory Data Analysis & Analytical Insights

Statistical profiling of the Amazon Fine Food Reviews dataset revealed critical customer behavior patterns:

| Metric / Dimension | Finding | Behavioral Insight |
| :--- | :--- | :--- |
| **Rating Skew** | **78.1%** rated 4 or 5 stars (~64% 5-star) | Customers are predominantly satisfied; satisfied buyers review at a 5:1 ratio over dissatisfied buyers. |
| **Review Length Profile** | **~350,000** reviews are under 70 words | Short-form praise ("Great taste, fast shipping!") dominates positive feedback. |
| **Mixed Opinion Length** | 2-star and 3-star reviews are longest | Conflicted buyers write longer paragraphs to articulate both positive and negative aspects. |
| **Negation Prominence** | `"not"` is the #1 word in 1-star reviews | Negations like *"not good"*, *"not fresh"*, and *"never buy"* drive negative classification. |
| **Helpfulness Ratio** | Median = 1.0 for 5-star reviews | Readers strongly agree on helpfulness of positive reviews; critical reviews exhibit higher variance. |

---

## 🧪 Model Methodology & Performance

### 1. Negation Preservation Rationale
Standard English stopword filters strip words like `not`, `no`, `never`, and `without`. In sentiment analysis, doing so destroys the semantic difference between *"good"* and *"not good"*. 

Our custom preprocessor preserves **16 sentiment-critical tokens**:
```python
PRESERVED_STOPWORDS = [
    "not", "no", "nor", "never", "none", "neither",
    "very", "too", "so", "more", "most", "only", "just",
    "but", "against", "without", "until", "if"
]
```

### 2. Multi-Class Balanced Logistic Regression
Because 5-star reviews account for nearly two-thirds of the dataset, unweighted models achieve high naive accuracy by simply predicting 5 stars for everything, missing unhappy customers. Applying `class_weight='balanced'` penalizes misclassification inversely proportional to class frequency:

$$\text{Weight}_c = \frac{N_{\text{total}}}{C \times N_c}$$

### 3. Evaluation Metrics (80/20 Stratified Split)

| Metric | Score |
| :--- | :--- |
| **Weighted Precision** | **77.8%** |
| **Multi-Class Accuracy** | **73.2%** |
| **Weighted Recall** | **73.0%** |
| **Feature Space** | 50,000 unigrams + bigrams |

---

## 💻 Streamlit Web Application

The application is structured into four intuitive modules:

1. **🔮 Real-Time Predictor**:
   - Type or paste any product review or choose from built-in sample reviews.
   - Outputs predicted star rating (1–5 ⭐), sentiment category badge (Positive 🟢 / Neutral 🟡 / Negative 🔴), and confidence percentage.
   - Interactive bar chart showing class probability distribution across all 5 star ratings.
   - **Feature Weight Inspector**: Highlights top TF-IDF n-grams influencing the model's decision and shows preserved negation tokens.

2. **📊 EDA & Insights Dashboard**:
   - Interactive Plotly visualizations for star distribution, rating percentage shares, review length histograms, and rating vs. helpfulness boxplots.
   - Top frequent words comparison chart (Positive vs. Negative reviews).

3. **📁 Batch Prediction & Export**:
   - Upload any CSV or Excel file containing customer reviews.
   - Run batch sentiment and star inference with progress tracking.
   - Filter predictions by sentiment or rating and export enriched results to CSV.

4. **🧠 Model Architecture & Technical Specs**:
   - Technical walkthrough of the preprocessing pipeline, TF-IDF feature space, and cost-sensitive balanced logistic formulation.

---

## 📂 Repository Structure

```
amazon-review-analysis/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI workflow
├── data/
│   ├── sample_reviews.csv       # Benchmark sample dataset
│   └── README.md                # Kaggle dataset setup instructions
├── models/
│   ├── sentiment_pipeline.joblib # Serialized model & vectorizer
│   └── README.md                # Model persistence details
├── notebooks/
│   └── amazon_review_analysis.ipynb # Original exploratory & modeling notebook
├── src/
│   ├── __init__.py              # Package initialization
│   ├── preprocessor.py          # Negation-preserving NLP cleaning & lemmatization
│   ├── model.py                 # TF-IDF + Logistic Regression pipeline
│   ├── visualizer.py            # Plotly interactive chart generators
│   └── utils.py                 # Sample data loader & helpers
├── tests/
│   ├── __init__.py
│   └── test_preprocessor.py     # Unit test suite
├── app.py                       # Streamlit web application
├── train.py                     # CLI training and evaluation script
├── requirements.txt             # Python dependencies
├── .gitignore                   # Version control ignore rules
└── README.md                    # Project documentation
```

---

## 🛠️ Installation & Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/velagapudirushi/amazon-review-analysis.git
cd amazon-review-analysis
```

### 2. Create and Activate a Virtual Environment
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🏋️ Training the Model via CLI

You can retrain the model on the sample dataset or on the full Kaggle `Reviews.csv`:

```bash
# Train on sample dataset
python train.py --data data/sample_reviews.csv --output models/sentiment_pipeline.joblib

# Train on full Kaggle dataset
python train.py --data data/Reviews.csv --max-features 50000 --output models/sentiment_pipeline.joblib
```

---

## 🧪 Running Unit Tests

Run the automated test suite using Python's built-in `unittest`:

```bash
python3 -m unittest discover tests
```

---

## 👤 Author

**Velagapudi Rushi**
- **Email**: [velagapudirushi@gmail.com](mailto:velagapudirushi@gmail.com)
- **GitHub**: [@velagapudirushi](https://github.com/velagapudirushi)
- **LinkedIn**: [linkedin.com/in/velagapudirushi](https://www.linkedin.com/in/velagapudirushi)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
