# 📰🔍 TruthLens — ML-Powered Fake News Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red?logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green)

> **TruthLens** is an end-to-end fake news detection web application powered by classical Machine Learning, Deep Learning (LSTM/Bi-LSTM), and Transformer models (BERT, DistilBERT). Built with Streamlit for an interactive, premium UI experience.

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://truthlensfakenewsdetection.streamlit.app/)

---

## ✨ Features

- 🧠 **5 ML Models** — SVM, LSTM, Bi-LSTM, BERT (bert-tiny), DistilBERT
- ✍️ **Text Input** — Paste any news article body for instant classification
- 🔗 **URL Scraper** — Fetch and analyze live news articles directly from URLs
- 📊 **Model Comparison** — Performance table, Confusion Matrix, ROC Curves
- 📈 **Training Metrics** — Epoch-wise loss & accuracy charts for deep learning models
- 📥 **PDF Reports** — Download a full verification report for every prediction
- 🎨 **Premium Dark UI** — Glassmorphism design with smooth animations

---

## 🤖 Models & Accuracy

| Model | Type | Accuracy |
|-------|------|----------|
| **DistilBERT** | Transformer | **95.00%** |
| **SVM** | Classical ML | **92.00%** |
| **Bi-LSTM** | Deep Learning | **72.00%** |
| **BERT** *(bert-tiny)* | Transformer | **70.00%** |
| **LSTM** | Deep Learning | **63.00%** |

> 🏆 **Overall App Average Accuracy: ~78.40%**

---

## 🗂️ Project Structure

```
TruthLens/
├── app.py                          # Main Streamlit application
├── custom_style.css                # Custom UI styles & animations
├── requirements.txt                # Python dependencies
├── .gitignore
│
├── nlp/
│   └── preprocess.py               # Text cleaning, tokenizer, lemmatization
│
├── scraper/
│   └── news_scraper.py             # newspaper3k + BeautifulSoup URL scraper
│
├── training/
│   ├── train_ml.py                 # SVM training (TF-IDF + sklearn)
│   ├── train_lstm.py               # LSTM & Bi-LSTM training (PyTorch)
│   ├── train_bert.py               # BERT & DistilBERT fine-tuning (HuggingFace)
│   └── download_or_generate_dataset.py
│
├── models/
│   ├── metrics.json                # Saved evaluation metrics for all models
│   ├── svm_model.pkl               # Trained SVM (gitignored - large file)
│   ├── tfidf_vectorizer.pkl        # TF-IDF vectorizer (gitignored)
│   ├── lstm_model.pt               # LSTM weights (gitignored)
│   ├── bilstm_model.pt             # Bi-LSTM weights (gitignored)
│   ├── bert_model/                 # Fine-tuned BERT (gitignored)
│   └── distilbert_model/           # Fine-tuned DistilBERT (gitignored)
│
└── dataset/
    ├── fake.csv                    # Fake news dataset (gitignored)
    └── real.csv                    # Real news dataset (gitignored)
```

---

## ⚙️ Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/Shashi0103/TruthLens.git
cd TruthLens
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download spaCy model
```bash
python -m spacy download en_core_web_sm
```

### 5. Download NLTK data
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### 6. Download dataset & train models
```bash
# Download dataset
python training/download_or_generate_dataset.py

# Train SVM
python training/train_ml.py

# Train LSTM & Bi-LSTM
python training/train_lstm.py

# Fine-tune BERT & DistilBERT
python training/train_bert.py
```

### 7. Run the app
```bash
streamlit run app.py
```

---

## 🧪 NLP Pipeline

Every article goes through a robust preprocessing pipeline before classification:

1. **HTML stripping** — removes tags and scripts
2. **URL & special character removal**
3. **Lowercasing**
4. **NLTK stopword removal**
5. **spaCy lemmatization** (`en_core_web_sm`)

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit + Custom CSS |
| Classical ML | scikit-learn (SVM + TF-IDF) |
| Deep Learning | PyTorch (LSTM, Bi-LSTM) |
| Transformers | HuggingFace (BERT, DistilBERT) |
| NLP | spaCy, NLTK |
| Scraping | newspaper3k, BeautifulSoup4 |
| Visualization | Plotly |
| Reports | ReportLab (PDF) |

---

## 👤 Author

**Shashi Kumar Sahu**  
Made with ❤️ 

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
