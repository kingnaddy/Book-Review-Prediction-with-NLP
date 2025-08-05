---
project: "Book Review Sentiment Analysis"
overview: |
  Implements and evaluates ML approaches to classify book reviews as positive or negative.
  Compares TF-IDF + Logistic Regression vs. Bi-LSTM with Word2Vec embeddings.
problem_description: |
  **Goal:** Predict whether a book review is positive.  
  **Input:** Raw text in the `Review` column.  
  **Output:** Binary label in the `Positive Review` column.  
  **Value:** Early forecasting of public reception; optimized marketing spend.
data_preparation_and_exploration:
  dataset:
    path: "data/bookReviewsData.csv"
    shape: [1973, 2]
  class_distribution:
    negative: 993
    positive: 980
  steps:
    - Lowercase conversion
    - Punctuation removal
    - Stop-word removal
    - Lemmatization
    - Tokenization
    - Word2Vec embedding
modeling_and_evaluation_plan:
  logistic_regression:
    tfidf:
      ngram_range: [1, 4]
      min_df: 2
      max_df: 0.8
    classifier:
      type: "LogisticRegression"
      C: 100
    validation:
      train_test_split: [0.8, 0.2]
      cross_val: 5-fold accuracy
    test_accuracy: 0.8239
  bidirectional_lstm:
    architecture:
      - Masking(mask_value=0., input_shape=[100,100])
      - Bidirectional(LSTM(128))
      - Dropout(0.5)
      - Dense(1, activation="sigmoid")
    embeddings: "Word2Vec"
    optimizer: "SGD(lr=0.01)"
    loss: "binary_crossentropy"
    metrics: [accuracy]
    training: { epochs: 50, validation_split: 0.2 }
visualizations_and_insights:
  - name: "Loss Curves"
    description: "Training vs. validation loss over epochs."
  - name: "Confusion Matrix"
    description: "True/False positives & negatives breakdown."
  - name: "Classification Report"
    description: "Precision, recall & F1-score per class."
summary_and_findings: |
  - **Baseline:** TF-IDF + LogisticRegression → 82.39% test accuracy.  
  - **Deep Model:** Bi-LSTM shows promise; further tuning may exceed baseline.
references_and_acknowledgements:
  - "Pedregosa et al., Scikit-learn: Machine Learning in Python (2011)"
  - "Řehůřek & Sojka, Gensim—Topic Modelling for Humans (2010)"
  - "Hochreiter & Schmidhuber, Long Short-Term Memory (1997)"
license: "MIT"
contribution_guidelines: |
  1. Fork the repository  
  2. Create a feature branch  
  3. Submit a pull request  
  _Please open an issue for major changes._
---

# 📚 Book Review Sentiment Analysis

> A comparison of classic NLP and deep-learning models to classify book reviews and drive marketing insights.

## 🚀 Project Overview

Implements and evaluates ML approaches to classify book reviews as positive or negative. Compares TF-IDF + Logistic Regression vs. Bi-LSTM with Word2Vec embeddings.

## 📝 Problem Description

- **Goal:** Predict whether a review is positive.  
- **Input:** Raw text (`Review` column).  
- **Output:** Binary label (`Positive Review`).  
- **Business Value:** Early forecasting of reception; optimize marketing spend.

## 🔍 Data Preparation & Exploration

- **Dataset:** `data/bookReviewsData.csv` (1973 reviews × 2 columns)  
- **Class Balance:** 993 negative, 980 positive  
- **Preprocessing Steps:**  
  1. Lowercase conversion  
  2. Punctuation removal  
  3. Stop-word removal  
  4. Lemmatization  
  5. Tokenization  
  6. Word2Vec embedding

## 🛠 Modeling & Evaluation Plan

### 1. Logistic Regression Pipeline

- **TF-IDF:** ngram_range [1,4], min_df=2, max_df=0.8  
- **Classifier:** LogisticRegression(C=100)  
- **Validation:** 80/20 split + 5-fold CV  
- **Test Accuracy:** 82.39%

### 2. Bidirectional LSTM

- **Architecture:** Masking → Bi-LSTM(128) → Dropout(0.5) → Dense(1)  
- **Embeddings:** Word2Vec  
- **Optimizer:** SGD(lr=0.01)  
- **Loss:** binary_crossentropy  
- **Metrics:** accuracy  
- **Training:** 50 epochs, 20% validation

## 📊 Visualizations & Insights

- **Loss Curves:** Detect under-/over-fitting  
- **Confusion Matrix:** Class-level error analysis  
- **Classification Report:** Precision/recall/F1 per class

## 🧰 Usage Instructions

1. **Clone** this repo  
2. **Install** dependencies:  
   ```bash
   pip install -r requirements.txt
