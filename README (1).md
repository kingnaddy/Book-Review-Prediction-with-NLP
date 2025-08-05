# 📚 Sentiment Classification of Book Reviews Using Bidirectional LSTM

## 🧠 Project Overview
This project implements a complete machine learning lifecycle focused on predicting whether a book review is positive using natural language processing (NLP) techniques. We built a custom sentiment classifier leveraging a Bidirectional LSTM network trained on a Word2Vec-embedded corpus. The goal was to demonstrate deep learning and NLP proficiency in a real-world scenario.

---

## ❗ Problem Description
With the rising volume of user-generated content, authors and publishers can benefit from automated sentiment analysis of reviews to:
- Predict public reception of new books
- Improve marketing strategies
- Identify underperforming titles early

**Label (Target)**: `Positive Review`  
**Type of Problem**: Supervised, Binary Classification  
**Features Used**: Raw textual data from the `Review` column

---

## 📊 Data Preparation & Exploration
- Dataset: Book Review Dataset
- Inspected for null values and duplicates
- Observed class distribution to detect imbalance
- Previewed raw reviews alongside sentiment labels
- Visualized class distribution with bar plots

---

## 🔎 Modeling & Evaluation Plan
### Preprocessing:
- Used Gensim’s `simple_preprocess` for:
  - Lowercasing
  - Tokenization
  - Punctuation & stop word removal
- Trained Word2Vec embeddings (100-dim vectors)
- Applied zero-padding to uniform input lengths (100x100)

### Model Architecture:
- **Bidirectional LSTM** using TensorFlow Keras:
  - Masking layer to ignore padding
  - Two stacked BiLSTM layers (64 units each)
  - Dense output with sigmoid activation

### Training:
- Optimizer: Stochastic Gradient Descent (SGD), LR = 0.1
- Loss: Binary Crossentropy
- Validation Split: 20%
- Fine-tuned number of epochs, dropout, and layer structure for better generalization

---

## 📈 Visualizations & Insights
- Class distribution plot revealed a slightly imbalanced dataset
- Word2Vec model captured semantically related terms (e.g., _“tasteless” → “bland”, “boring”_)
- PCA/TSNE plots (if included in notebook) can further show vector space clustering of sentiments

---

## 🚀 Usage Instructions

### 1. Clone the repository:
```bash
git clone https://github.com/your-username/book-review-sentiment.git
cd book-review-sentiment
```

### 2. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the notebook:
Open `DefineAndSolveMLProblem.ipynb` in Jupyter Notebook and execute cells sequentially.

---

## 📌 Summary & Findings
- Successfully built an end-to-end sentiment classification model from raw reviews.
- Word2Vec + BiLSTM showed strong capability to capture semantic patterns in text.
- The project illustrates NLP preprocessing, word embeddings, and sequence modeling techniques.
- Future work: Incorporate attention mechanisms or transformer-based models for improved accuracy.

---

## 📚 References & Acknowledgements
- Gensim Word2Vec Documentation
- TensorFlow Keras LSTM Guide
- Dataset: Provided via course materials
- Special thanks to instructors and peers for feedback and guidance

---

## 📄 License & Contribution

### License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

### Contributions
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
