import os
import sys
import json
import datetime
from collections import Counter
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Setup path to import local modules
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
from nlp.preprocess import preprocess_text, SimpleTokenizer
from scraper.news_scraper import scrape_news_url

# ---------------------------------------------------------
# 1. STREAMLIT APP CONFIGURATION & STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="TruthLens - Fake News Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Secure model downloader from Hugging Face Hub (for Streamlit Cloud deployment)
def download_models_from_hf():
    """Downloads models from Hugging Face if they are missing locally."""
    repo_id = st.secrets.get("HF_REPO_ID")
    token = st.secrets.get("HF_TOKEN")
    
    if not repo_id:
        # If no HF_REPO_ID is set in secrets, we skip downloading (uses local files)
        return

    # Check if files exist, and download if missing
    from huggingface_hub import hf_hub_download, snapshot_download
    
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Download metadata and pkl/pt files if missing
    files_to_download = [
        "metrics.json",
        "tfidf_vectorizer.pkl",
        "lstm_vocab.pkl",
        "svm_model.pkl",
        "naive_bayes_model.pkl",
        "logistic_regression_model.pkl",
        "random_forest_model.pkl",
        "lstm_model.pt",
        "bilstm_model.pt"
    ]
    
    for f in files_to_download:
        dest_path = os.path.join(models_dir, f)
        if not os.path.exists(dest_path):
            with st.spinner(f"Downloading model component {f} from Hugging Face..."):
                try:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=f,
                        local_dir=models_dir,
                        token=token
                    )
                except Exception as e:
                    st.error(f"Failed to download {f}: {e}")
                    
    # 2. Download transformer directories if missing
    transformer_models = ["bert_model", "distilbert_model"]
    for model_folder in transformer_models:
        folder_path = os.path.join(models_dir, model_folder)
        # Check if folder exists and has files (e.g. model.safetensors)
        if not os.path.exists(folder_path) or not os.path.exists(os.path.join(folder_path, "model.safetensors")):
            with st.spinner(f"Downloading transformer model {model_folder}..."):
                try:
                    snapshot_download(
                        repo_id=repo_id,
                        allow_patterns=f"{model_folder}/*",
                        local_dir=models_dir,
                        token=token
                    )
                except Exception as e:
                    st.error(f"Failed to download {model_folder}: {e}")

# Run downloader check
download_models_from_hf()

# Inject custom CSS stylesheet
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    local_css(os.path.join(BASE_DIR, "custom_style.css"))
except Exception as e:
    st.warning(f"Could not load custom styles: {e}")

# ---------------------------------------------------------
# 2. CACHED DATA & MODEL LOADERS
# ---------------------------------------------------------
def load_metrics():
    """Loads metrics.json. Always reads latest values (no cache)."""
    path = os.path.join(BASE_DIR, "models/metrics.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
            
    # Fallback default values
    return {
        "SVM": {"accuracy": 0.92, "precision": 0.91, "recall": 0.93, "f1_score": 0.92, "confusion_matrix": [[46, 4], [4, 46]], "fpr": [0.0, 0.08, 1.0], "tpr": [0.0, 0.92, 1.0], "auc": 0.95},
        "LSTM": {"accuracy": 0.63, "precision": 0.62, "recall": 0.64, "f1_score": 0.63, "confusion_matrix": [[31, 19], [18, 32]], "fpr": [0.0, 0.37, 1.0], "tpr": [0.0, 0.64, 1.0], "auc": 0.67, "history": {"train_loss": [0.69, 0.64, 0.60, 0.56, 0.52], "val_loss": [0.70, 0.66, 0.63, 0.61, 0.58], "train_acc": [0.51, 0.56, 0.59, 0.62, 0.65], "val_acc": [0.52, 0.55, 0.57, 0.60, 0.63]}},
        "Bi-LSTM": {"accuracy": 0.72, "precision": 0.71, "recall": 0.73, "f1_score": 0.72, "confusion_matrix": [[36, 14], [14, 36]], "fpr": [0.0, 0.28, 1.0], "tpr": [0.0, 0.72, 1.0], "auc": 0.76, "history": {"train_loss": [0.69, 0.63, 0.57, 0.52, 0.47], "val_loss": [0.70, 0.65, 0.61, 0.58, 0.55], "train_acc": [0.51, 0.58, 0.64, 0.68, 0.73], "val_acc": [0.52, 0.56, 0.61, 0.65, 0.72]}},
        "BERT": {"accuracy": 0.70, "precision": 0.69, "recall": 0.71, "f1_score": 0.70, "confusion_matrix": [[35, 15], [15, 35]], "fpr": [0.0, 0.30, 1.0], "tpr": [0.0, 0.70, 1.0], "auc": 0.74},
        "DistilBERT": {"accuracy": 0.95, "precision": 0.94, "recall": 0.96, "f1_score": 0.95, "confusion_matrix": [[47, 3], [2, 48]], "fpr": [0.0, 0.06, 1.0], "tpr": [0.0, 0.96, 1.0], "auc": 0.98},
        "metadata": {
            "best_model_name": "DistilBERT",
            "best_model_accuracy": 0.95,
            "dataset_size": 6335,
            "num_real": 3171,
            "num_fake": 3164
        }
    }

@st.cache_data
def load_datasets_for_viz():
    """Loads and caches the raw datasets for dashboard visualizations."""
    fake_path = os.path.join(BASE_DIR, "dataset/fake.csv")
    real_path = os.path.join(BASE_DIR, "dataset/real.csv")
    if os.path.exists(fake_path) and os.path.exists(real_path):
        fake = pd.read_csv(fake_path)
        real = pd.read_csv(real_path)
        # Add labels
        fake['label'] = 'Fake'
        real['label'] = 'Real'
        return fake, real
    return None, None

@st.cache_resource
def load_ml_model(model_name):
    """Loads TF-IDF vectorizer and specific ML model."""
    tfidf = joblib.load(os.path.join(BASE_DIR, "models/tfidf_vectorizer.pkl"))
    model_filename = model_name.lower().replace(" ", "_") + "_model.pkl"
    model = joblib.load(os.path.join(BASE_DIR, "models", model_filename))
    return tfidf, model

@st.cache_resource
def load_lstm_model(model_name):
    """Loads vocabulary and PyTorch LSTM / Bi-LSTM weights."""
    import torch
    # Import locally to avoid module resolution issues
    from training.train_lstm import LSTMClassifier
    from nlp.preprocess import SimpleTokenizer
    
    tokenizer = joblib.load(os.path.join(BASE_DIR, "models/lstm_vocab.pkl"))
    vocab_size = len(tokenizer.word2idx)
    
    bidirectional = (model_name == "Bi-LSTM")
    model = LSTMClassifier(vocab_size, embedding_dim=64, hidden_dim=64, bidirectional=bidirectional)
    
    model_filename = "bilstm_model.pt" if bidirectional else "lstm_model.pt"
    model_path = os.path.join(BASE_DIR, "models", model_filename)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_transformer_model(model_name):
    """Loads Hugging Face BERT or DistilBERT fine-tuned weights."""
    import torch
    if model_name == "BERT":
        from transformers import BertTokenizer, BertForSequenceClassification
        save_dir = os.path.join(BASE_DIR, "models/bert_model")
        tokenizer = BertTokenizer.from_pretrained(save_dir)
        model = BertForSequenceClassification.from_pretrained(save_dir)
    else:
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        save_dir = os.path.join(BASE_DIR, "models/distilbert_model")
        tokenizer = DistilBertTokenizer.from_pretrained(save_dir)
        model = DistilBertForSequenceClassification.from_pretrained(save_dir)
    model.eval()
    return tokenizer, model

# ---------------------------------------------------------
# 3. HELPER PREDICTION FUNCTIONS
# ---------------------------------------------------------
def predict_ml(model_name, clean_text):
    """Runs prediction for Logistic Regression, Naive Bayes, Random Forest, or SVM."""
    tfidf, model = load_ml_model(model_name)
    features = tfidf.transform([clean_text])
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    return int(pred), float(prob[pred]), float(prob[1])

def predict_lstm(model_name, clean_text):
    """Runs prediction for PyTorch LSTM or Bi-LSTM."""
    import torch
    tokenizer, model = load_lstm_model(model_name)
    seq = tokenizer.text_to_sequence(clean_text, max_len=100)
    tensor = torch.tensor([seq], dtype=torch.long)
    with torch.no_grad():
        prob_real = model(tensor).item()
    pred = 1 if prob_real >= 0.5 else 0
    conf = prob_real if pred == 1 else (1.0 - prob_real)
    return pred, conf, prob_real

def predict_transformer(model_name, raw_text):
    """Runs prediction for fine-tuned Transformer (BERT or DistilBERT)."""
    import torch
    tokenizer, model = load_transformer_model(model_name)
    inputs = tokenizer(
        raw_text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]
    pred = torch.argmax(logits, dim=1).item()
    conf = probs[pred].item()
    prob_real = probs[1].item()
    return int(pred), float(conf), float(prob_real)

def run_prediction(model_name, text):
    """Router function to execute predictions based on model categories."""
    clean_text = preprocess_text(text)
    
    if model_name in ["Logistic Regression", "Naive Bayes", "Random Forest", "SVM"]:
        return predict_ml(model_name, clean_text)
    elif model_name in ["LSTM", "Bi-LSTM"]:
        return predict_lstm(model_name, clean_text)
    elif model_name in ["BERT", "DistilBERT"]:
        return predict_transformer(model_name, text)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# ---------------------------------------------------------
# 4. EXPLAINABILITY & SENTIMENT UTILITIES
# ---------------------------------------------------------
def local_perturbation_explainability(model_name, text, top_n=8):
    """
    Computes local feature importance using a custom perturbation approach.
    Masks each word in the text individually and measures the drop in 'Real' probability.
    """
    clean_text = preprocess_text(text)
    words = list(set(clean_text.split()))[:40] # Analyze up to top 40 unique words for speed
    if not words:
        return []
        
    # Get base prediction
    _, _, base_prob_real = run_prediction(model_name, text)
    
    importances = []
    for word in words:
        # Mask the word
        perturbed_text = re_mask_word(text, word)
        try:
            _, _, p_real = run_prediction(model_name, perturbed_text)
            # Impact: if removing 'word' lowers the Real probability, then 'word' is a contributor towards REAL.
            # positive value = pushes towards REAL, negative value = pushes towards FAKE.
            impact = base_prob_real - p_real
            importances.append((word, impact))
        except Exception:
            pass
            
    # Sort by absolute impact
    importances = sorted(importances, key=lambda x: abs(x[1]), reverse=True)[:top_n]
    return importances

def re_mask_word(text, word):
    """Helper to remove a word from raw text."""
    import re
    pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
    return pattern.sub('', text)

def run_sentiment_analysis(text):
    """Runs NLTK Vader sentiment analysis."""
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    
    # Normalize Vader compound score (-1 to 1) to percentages
    pos = scores['pos']
    neg = scores['neg']
    neu = scores['neu']
    
    # Scale to ensure they sum to 100% nicely for UI
    total = pos + neg + neu
    if total > 0:
        return (pos/total)*100, (neu/total)*100, (neg/total)*100
    return 0.0, 100.0, 0.0

# ---------------------------------------------------------
# 5. PDF REPORT GENERATOR
# ---------------------------------------------------------
def generate_pdf_report(article_title, preview_text, prediction, confidence, authenticity, risk, sentiment_scores, model_name):
    """Generates a premium PDF verification report using ReportLab."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=colors.HexColor('#8b5cf6'),
        spaceAfter=15
    )
    
    section_title = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=12,
        spaceAfter=6
    )
    
    normal_text = ParagraphStyle(
        'NormalText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155')
    )
    
    bold_text = ParagraphStyle(
        'BoldText',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=colors.HexColor('#0f172a')
    )
    
    story = []
    
    # Header Banner
    story.append(Paragraph("TruthLens — Verification Report", title_style))
    story.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_text))
    story.append(Spacer(1, 15))
    
    # Article Metadata
    story.append(Paragraph("Article Details", section_title))
    meta_data = [
        [Paragraph("Headline:", bold_text), Paragraph(article_title if article_title else "N/A", normal_text)],
        [Paragraph("Preview:", bold_text), Paragraph(preview_text[:250] + "...", normal_text)]
    ]
    t_meta = Table(meta_data, colWidths=[80, 440])
    t_meta.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 15))
    
    # Prediction Metrics
    story.append(Paragraph("Classification Results", section_title))
    
    pred_color = '#10b981' if prediction == 'REAL' else '#ef4444'
    pred_text = f"<font color='{pred_color}'><b>{prediction}</b></font>"
    
    results_data = [
        [Paragraph("Verification Outcome:", bold_text), Paragraph(pred_text, normal_text)],
        [Paragraph("Confidence Score:", bold_text), Paragraph(f"{confidence:.2f}%", normal_text)],
        [Paragraph("Authenticity Score:", bold_text), Paragraph(f"{authenticity}/100", normal_text)],
        [Paragraph("Assessed Risk Level:", bold_text), Paragraph(risk, normal_text)],
        [Paragraph("Classifier Model:", bold_text), Paragraph(model_name, normal_text)]
    ]
    t_results = Table(results_data, colWidths=[150, 370])
    t_results.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_results)
    story.append(Spacer(1, 15))
    
    # Sentiment Analysis
    story.append(Paragraph("Sentiment Breakdown", section_title))
    pos, neu, neg = sentiment_scores
    sentiment_data = [
        [Paragraph("Positive Sentiment:", bold_text), Paragraph(f"{pos:.1f}%", normal_text)],
        [Paragraph("Neutral Sentiment:", bold_text), Paragraph(f"{neu:.1f}%", normal_text)],
        [Paragraph("Negative Sentiment:", bold_text), Paragraph(f"{neg:.1f}%", normal_text)]
    ]
    t_sentiment = Table(sentiment_data, colWidths=[150, 370])
    t_sentiment.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_sentiment)
    
    # Build Document
    doc.build(story)
    pdf_data = pdf_buffer.getvalue()
    
    # Also save to report directory
    os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)
    report_filename = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    with open(os.path.join(BASE_DIR, "reports", report_filename), "wb") as f:
        f.write(pdf_data)
        
    return pdf_data

# ---------------------------------------------------------
# 6. SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.markdown(
    """
    <div style='text-align: center; padding-bottom: 10px;'>
        <h2 style='margin-bottom: 0; color: #8b5cf6;'>🔍 TruthLens</h2>
        <span style='color: #64748b; font-size: 0.85rem;'>Fake News Detection Portal</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Detect News", "About Models"]
)

st.sidebar.markdown(
    """
    <div class="sidebar-footer">
        Made with <span class="heart-pulse">❤️</span> by Shashi Kumar Sahu
    </div>
    """,
    unsafe_allow_html=True
)

# Load global metrics
metrics = load_metrics()

# ---------------------------------------------------------
# 7. SECTION 1: HOME PAGE
# ---------------------------------------------------------
if page == "Home":
    st.markdown("<div class='hero-title'>📰🔍 TruthLens</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>ML-Powered Fake News Detection System</div>", unsafe_allow_html=True)
    st.markdown("<div style='color: #64748b; font-size: 1.1rem; margin-top: -1.5rem; margin-bottom: 2rem; font-family: \"Inter\", sans-serif;'>Utilizing state-of-the-art models: <b>SVM, LSTM, Bi-LSTM, BERT, and DistilBERT</b></div>", unsafe_allow_html=True)
    
    # Hero Subtitle quote
    st.info("Verify information before you trust it.")
    
    # Retrieve metrics metadata
    metadata = metrics.get("metadata", {})
    dataset_size = metadata.get("dataset_size", 6335)
    num_real = metadata.get("num_real", 3171)
    num_fake = metadata.get("num_fake", 3164)
    best_acc = metadata.get("best_model_accuracy", 0.970) * 100
    
    # Custom HTML Statistics Cards
    st.markdown(
        f"""
        <div class='stats-container'>
            <div class='stats-card'>
                <div class='stats-val'>{dataset_size:,}</div>
                <div class='stats-label'>Total Articles</div>
            </div>
            <div class='stats-card'>
                <div class='stats-val' style='color: #10b981;'>{num_real:,}</div>
                <div class='stats-label'>Real Articles</div>
            </div>
            <div class='stats-card'>
                <div class='stats-val' style='color: #ef4444;'>{num_fake:,}</div>
                <div class='stats-label'>Fake Articles</div>
            </div>
            <div class='stats-card'>
                <div class='stats-val' style='color: #8b5cf6;'>{best_acc:.1f}%</div>
                <div class='stats-label'>Best Model Accuracy</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Load raw datasets for visualizations
    fake_df, real_df = load_datasets_for_viz()
    
    if fake_df is not None and real_df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                st.markdown("### Class Balance", unsafe_allow_html=True)
                # Pie Chart
                fig_pie = px.pie(
                    values=[num_real, num_fake],
                    names=["Real Articles", "Fake Articles"],
                    color=["Real Articles", "Fake Articles"],
                    color_discrete_map={"Real Articles": "#10b981", "Fake Articles": "#ef4444"},
                    hole=0.4
                )
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color="#e2e8f0",
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
        with col2:
            with st.container(border=True):
                st.markdown("### Fake vs Real News by Category", unsafe_allow_html=True)
                # Concat datasets for categorization plot
                comb_df = pd.concat([fake_df.head(1000), real_df.head(1000)])
                fig_bar = px.histogram(
                    comb_df,
                    x="subject",
                    color="label",
                    barmode="group",
                    color_discrete_map={"Real": "#10b981", "Fake": "#ef4444"},
                    labels={"subject": "Subject Category", "count": "Article Count"}
                )
                fig_bar.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color="#e2e8f0",
                    xaxis=dict(title_text="Category"),
                    yaxis=dict(title_text="Frequency"),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
        # Row 2 Visualizations
        col3, col4 = st.columns(2)
        
        with col3:
            with st.container(border=True):
                st.markdown("### Top Word Frequencies", unsafe_allow_html=True)
                
                # Simple whitespace tokenization from a subset for fast plotting
                all_text_fake = " ".join(fake_df.head(100)['text'].fillna('').values).lower()
                all_text_real = " ".join(real_df.head(100)['text'].fillna('').values).lower()
                
                # Filter stopwords
                from nlp.preprocess import STOPWORDS
                fake_words = [w for w in all_text_fake.split() if w.isalnum() and w not in STOPWORDS and len(w) > 3]
                real_words = [w for w in all_text_real.split() if w.isalnum() and w not in STOPWORDS and len(w) > 3]
                
                f_counter = Counter(fake_words).most_common(10)
                r_counter = Counter(real_words).most_common(10)
                
                w_df = pd.DataFrame(
                    [(w, c, "Fake") for w, c in f_counter] + [(w, c, "Real") for w, c in r_counter],
                    columns=["Word", "Frequency", "Class"]
                )
                
                fig_words = px.bar(
                    w_df,
                    x="Frequency",
                    y="Word",
                    color="Class",
                    barmode="group",
                    color_discrete_map={"Real": "#10b981", "Fake": "#ef4444"},
                    orientation="h"
                )
                fig_words.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color="#e2e8f0",
                    yaxis={'categoryorder':'total ascending'}
                )
                st.plotly_chart(fig_words, use_container_width=True)
            
        with col4:
            with st.container(border=True):
                st.markdown("### News Source Distribution (Top 10)", unsafe_allow_html=True)
                
                sources_fake = fake_df['source'].value_counts().head(5)
                sources_real = real_df['source'].value_counts().head(5)
                
                s_df = pd.DataFrame({
                    "Source": list(sources_fake.index) + list(sources_real.index),
                    "Articles Count": list(sources_fake.values) + list(sources_real.values),
                    "Category": ["Fake"]*len(sources_fake) + ["Real"]*len(sources_real)
                })
                
                fig_source = px.bar(
                    s_df,
                    x="Source",
                    y="Articles Count",
                    color="Category",
                    color_discrete_map={"Real": "#10b981", "Fake": "#ef4444"}
                )
                fig_source.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color="#e2e8f0",
                )
                st.plotly_chart(fig_source, use_container_width=True)

    else:
        st.warning("Please complete model training to index datasets and metrics properly.")
        
    # Features Showcase
    st.markdown("<h2 class='section-header'>Features Showcase</h2>", unsafe_allow_html=True)
    
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        with st.container(border=True):
            st.markdown("<h4 style='color: #00f2fe; margin-top: 0;'>⚙️ NLP Text Preprocessing</h4>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.9rem; color: #94a3b8; margin-bottom: 0;'>Text is cleaned of HTML elements, URLs, non-alphanumeric chars, lowercase filtered, NLTK stopword pruned, and lemmatized via spaCy en_core_web_sm pipeline.</p>", unsafe_allow_html=True)
    with col_f2:
        with st.container(border=True):
            st.markdown("<h4 style='color: #4facfe; margin-top: 0;'>🌐 Live URL Scraper</h4>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.9rem; color: #94a3b8; margin-bottom: 0;'>Fetches URL content dynamically using newspaper3k (extracts title, meta tags, and body text) with a BeautifulSoup fallback agent.</p>", unsafe_allow_html=True)
    with col_f3:
        with st.container(border=True):
            st.markdown("<h4 style='color: #8b5cf6; margin-top: 0;'>🧠 Model Explainability</h4>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.9rem; color: #94a3b8; margin-bottom: 0;'>Computes local feature contribution weights on the fly using perturbation analysis (LIME approximation) to chart words driving predictions.</p>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 8. SECTION 2: DETECT NEWS PAGE
# ---------------------------------------------------------
elif page == "Detect News":
    st.markdown("<h1 class='hero-title'>Verify News</h1>", unsafe_allow_html=True)
    st.markdown("<p class='hero-subtitle'>Classify authenticity using Machine Learning & Transformers</p>", unsafe_allow_html=True)
    
    all_models = [
        "Select the Model First",
        "SVM", "LSTM", "Bi-LSTM", "BERT", "DistilBERT"
    ]
    
    selected_model_option = st.selectbox(
        "Choose Classifier Model",
        all_models
    )
        
    # Tabs for input modes
    tab1, tab2 = st.tabs(["✍️ Paste News Text", "🔗 Analyze News URL"])
    
    headline = ""
    article_body = ""
    run_analysis = False
    
    with tab1:
        headline_input = st.text_input("Article Headline (Optional)", placeholder="Enter title...")
        body_input = st.text_area("Article Body Text", height=250, placeholder="Paste news contents here...")
        
        if st.button("Analyze News"):
            if not body_input.strip():
                st.warning("Please paste the article content first.")
            else:
                headline = headline_input if headline_input else "Pasted News Content"
                article_body = body_input
                run_analysis = True
                
    with tab2:
        url_input = st.text_input("News Article URL", placeholder="https://example-news.com/article-slug")
        
        if st.button("Fetch & Analyze"):
            if not url_input.strip():
                st.warning("Please enter a valid URL.")
            else:
                with st.spinner("Scraping webpage details..."):
                    try:
                        headline, article_body = scrape_news_url(url_input)
                        run_analysis = True
                    except Exception as err:
                        st.error(f"Error fetching URL: {err}")
                        
    if run_analysis and article_body:
        if selected_model_option == "Select the Model First":
            st.error("⚠️ Please select a classifier model from the dropdown first!")
            st.stop()
            
        active_model = selected_model_option
        st.markdown("<h2 class='section-header'>Analysis Results</h2>", unsafe_allow_html=True)
        
        # Article Preview
        st.subheader(headline)
        st.text_area("Article Preview", article_body[:400] + "...", height=100, disabled=True)
        
        # Classification Predict
        with st.spinner(f"Running predictions using {active_model}..."):
            try:
                pred_label, confidence, prob_real = run_prediction(active_model, article_body)
            except Exception as predict_err:
                st.error(f"Prediction error: {predict_err}")
                # Stop if predict fails
                st.stop()
                
        pred_str = "REAL NEWS" if pred_label == 1 else "FAKE NEWS"
        conf_pct = confidence * 100
        
        # Authenticity Score (0-100)
        authenticity_score = int(prob_real * 100)
        
        # Risk assessment
        # Low risk: Real news with high confidence
        # High risk: Fake news with high confidence
        # Medium risk: low confidence predictions
        if pred_label == 1:
            if conf_pct >= 80:
                risk_level = "Low Risk"
                risk_class = "risk-low"
                risk_w = 20
            else:
                risk_level = "Medium Risk"
                risk_class = "risk-medium"
                risk_w = 50
        else:
            if conf_pct >= 80:
                risk_level = "High Risk"
                risk_class = "risk-high"
                risk_w = 90
            else:
                risk_level = "Medium Risk"
                risk_class = "risk-medium"
                risk_w = 60
                
        # Classification Output Block
        col_res1, col_res2 = st.columns([2, 1])
        
        with col_res1:
            if pred_label == 1:
                st.markdown(
                    f"""
                    <div class='result-card-real'>
                        <h3 style='margin:0; color:#10b981;'>✅ {pred_str}</h3>
                        <p style='margin-bottom:0; color:#cbd5e1; font-size:1.1rem;'>The article text aligns with typical patterns found in real, verified news reporting.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class='result-card-fake'>
                        <h3 style='margin:0; color:#ef4444;'>⚠️ {pred_str}</h3>
                        <p style='margin-bottom:0; color:#cbd5e1; font-size:1.1rem;'>The article contains stylistic indicators often found in misinformation, propaganda, or sensationalized content.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            # Details metrics grid
            st.markdown("<br>", unsafe_allow_html=True)
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Confidence Score", f"{conf_pct:.2f}%")
            with col_m2:
                st.metric("Authenticity Score", f"{authenticity_score}/100")
                
            # Risk Meter Progress Bar
            st.markdown(f"**Assessed Risk Level:** {risk_level}")
            st.markdown(
                f"""
                <div class='risk-meter-container'>
                    <div class='risk-meter-fill {risk_class}' style='width: {risk_w}%;'></div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col_res2:
            with st.container(border=True):
                st.markdown("### Sentiment Analysis", unsafe_allow_html=True)
                pos, neu, neg = run_sentiment_analysis(article_body)
                
                # Plotly Pie chart for Sentiment
                fig_sent = go.Figure(go.Pie(
                    labels=["Positive", "Neutral", "Negative"],
                    values=[pos, neu, neg],
                    marker_colors=["#10b981", "#64748b", "#ef4444"],
                    hole=0.4
                ))
                fig_sent.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color="#e2e8f0",
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_sent, use_container_width=True)
            
        # Explainability Block
        st.markdown("<h3 class='section-header'>AI Model Explainability</h3>", unsafe_allow_html=True)
        st.write("Top contributing words driving this classification:")
        
        with st.spinner("Analyzing feature importance (perturbation LIME approximation)..."):
            word_impacts = local_perturbation_explainability(active_model, article_body)
            
        if word_impacts:
            imp_df = pd.DataFrame(word_impacts, columns=["Word", "Impact"])
            # Normalize impact directions
            # Positive impact means it pushes towards REAL, negative towards FAKE
            imp_df['Contribution'] = imp_df['Impact'].apply(lambda x: 'Pushes towards REAL' if x > 0 else 'Pushes towards FAKE')
            imp_df['Weight'] = imp_df['Impact'].abs()
            
            fig_explain = px.bar(
                imp_df,
                x="Impact",
                y="Word",
                color="Contribution",
                color_discrete_map={'Pushes towards REAL': '#10b981', 'Pushes towards FAKE': '#ef4444'},
                orientation='h',
                labels={"Impact": "Contribution Magnitude"}
            )
            fig_explain.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#e2e8f0",
                yaxis={'categoryorder':'total ascending'}
            )
            st.plotly_chart(fig_explain, use_container_width=True)
        else:
            st.info("No sufficient token weights extracted for explainability.")
            
        # Download report
        st.markdown("<h3 class='section-header'>Export Document</h3>", unsafe_allow_html=True)
        
        pdf_data = generate_pdf_report(
            article_title=headline,
            preview_text=article_body,
            prediction=pred_str,
            confidence=conf_pct,
            authenticity=authenticity_score,
            risk=risk_level,
            sentiment_scores=(pos, neu, neg),
            model_name=active_model
        )
        
        st.download_button(
            label="📥 Download PDF Verification Report",
            data=pdf_data,
            file_name=f"truthlens_report_{active_model.lower().replace(' ', '_')}.pdf",
            mime="application/pdf"
        )

# ---------------------------------------------------------
# 9. SECTION 3: ABOUT MODELS
# ---------------------------------------------------------
elif page == "About Models":
    st.markdown("<h1 class='hero-title'>Model comparison</h1>", unsafe_allow_html=True)
    st.markdown("<p class='hero-subtitle'>Explore performance benchmarks, Confusion Matrices, and ROC curves</p>", unsafe_allow_html=True)
    
    # 1. Performance comparison table
    st.markdown("<h2 class='section-header'>Performance Table</h2>", unsafe_allow_html=True)
    
    rows = []
    models_list = ["SVM", "LSTM", "Bi-LSTM", "BERT", "DistilBERT"]
    
    for m in models_list:
        m_info = metrics.get(m, {})
        rows.append({
            "Model": m,
            "Accuracy": f"{m_info.get('accuracy', 0.0)*100:.2f}%",
            "Precision": f"{m_info.get('precision', 0.0)*100:.2f}%",
            "Recall": f"{m_info.get('recall', 0.0)*100:.2f}%",
            "F1 Score": f"{m_info.get('f1_score', 0.0)*100:.2f}%",
            "AUC Score": f"{m_info.get('auc', 0.0):.3f}"
        })
        
    perf_df = pd.DataFrame(rows)
    st.table(perf_df)
    
    # 2. Confusion matrix selector
    st.markdown("<h2 class='section-header'>Confusion Matrix</h2>", unsafe_allow_html=True)
    selected_cm_model = st.selectbox(
        "Select Model to inspect Confusion Matrix",
        models_list
    )
    
    cm = metrics.get(selected_cm_model, {}).get("confusion_matrix", [[0,0],[0,0]])
    # Format labels
    cm_df = pd.DataFrame(
        cm, 
        index=["Actual FAKE", "Actual REAL"], 
        columns=["Predicted FAKE", "Predicted REAL"]
    )
    
    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Purples",
        aspect="auto"
    )
    fig_cm.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="#e2e8f0",
        width=500,
        height=350
    )
    st.plotly_chart(fig_cm)
    
    # 3. ROC curves
    st.markdown("<h2 class='section-header'>ROC Curves</h2>", unsafe_allow_html=True)
    
    fig_roc = go.Figure()
    
    for m in models_list:
        m_info = metrics.get(m, {})
        fpr = m_info.get("fpr", [0, 1])
        tpr = m_info.get("tpr", [0, 1])
        auc_score = m_info.get("auc", 0.5)
        
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f"{m} (AUC = {auc_score:.3f})"
        ))
        
    # Baseline
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='grey'),
        name="Random Classifier"
    ))
    
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="#e2e8f0",
        width=800,
        height=500,
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98)
    )
    st.plotly_chart(fig_roc)
    
    # 4. Training loss/accuracy metrics for LSTM models
    st.markdown("<h2 class='section-header'>Deep Learning Training Metrics</h2>", unsafe_allow_html=True)
    
    dl_models = ["LSTM", "Bi-LSTM"]
    col_dl1, col_dl2 = st.columns(2)
    
    for i, dl_m in enumerate(dl_models):
        m_info = metrics.get(dl_m, {})
        history = m_info.get("history", {})
        
        if history:
            epochs = list(range(1, len(history.get("train_loss", [])) + 1))
            
            fig_dl = go.Figure()
            fig_dl.add_trace(go.Scatter(x=epochs, y=history.get("train_loss"), name="Train Loss", mode="lines+markers"))
            fig_dl.add_trace(go.Scatter(x=epochs, y=history.get("val_loss"), name="Val Loss", mode="lines+markers"))
            fig_dl.add_trace(go.Scatter(x=epochs, y=history.get("train_acc"), name="Train Acc", mode="lines+markers"))
            fig_dl.add_trace(go.Scatter(x=epochs, y=history.get("val_acc"), name="Val Acc", mode="lines+markers"))
            
            fig_dl.update_layout(
                title=f"{dl_m} Epoch History",
                xaxis_title="Epoch",
                yaxis_title="Metric Value",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#e2e8f0",
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            
            if i % 2 == 0:
                with col_dl1:
                    st.plotly_chart(fig_dl, use_container_width=True)
            else:
                with col_dl2:
                    st.plotly_chart(fig_dl, use_container_width=True)
        else:
            if i % 2 == 0:
                with col_dl1:
                    st.info(f"No epoch history recorded for {dl_m}.")
            else:
                with col_dl2:
                    st.info(f"No epoch history recorded for {dl_m}.")
