import os
import sys
import argparse
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc

# Add project root to python path to import nlp/preprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nlp.preprocess import preprocess_text

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

def load_data(quick_mode=True):
    print("Loading data from CSVs...")
    fake_df = pd.read_csv(os.path.join(BASE_DIR, "dataset/fake.csv"))
    real_df = pd.read_csv(os.path.join(BASE_DIR, "dataset/real.csv"))
    
    # Label them
    fake_df['label'] = 0
    real_df['label'] = 1
    
    if quick_mode:
        print("Quick mode enabled: Subsampling 500 fake and 500 real articles...")
        fake_df = fake_df.sample(min(500, len(fake_df)), random_state=42)
        real_df = real_df.sample(min(500, len(real_df)), random_state=42)
        
    df = pd.concat([fake_df, real_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", default=True, help="Train on a small subset for fast verification")
    parser.add_argument("--full", action="store_false", dest="quick", help="Train on the full dataset")
    args = parser.parse_args()
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Load and Clean data
    df = load_data(quick_mode=args.quick)
    
    print("Pre-processing text articles (HTML cleaning, lemmatization)...")
    # Clean text (combining headline + body for richer features)
    df['combined_text'] = df['title'].fillna('') + " " + df['text'].fillna('')
    df['clean_text'] = df['combined_text'].apply(preprocess_text)
    
    # Remove empty rows after preprocessing
    df = df[df['clean_text'].str.strip() != ''].reset_index(drop=True)
    
    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # 3. TF-IDF Vectorization
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Save TF-IDF Vectorizer
    joblib.dump(vectorizer, TFIDF_PATH)
    print(f"Saved TF-IDF Vectorizer to {TFIDF_PATH}")
    
    # 4. Define Traditional ML models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM": SVC(kernel='linear', probability=True, random_state=42)
    }
    
    # Structure for loading metrics
    metrics_data = {}
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                metrics_data = json.load(f)
        except Exception:
            metrics_data = {}
            
    best_accuracy = 0.0
    best_model_name = ""
    
    # 5. Train and Evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_tfidf, y_train)
        
        # Predictions & Probabilities
        y_pred = model.predict(X_test_tfidf)
        y_prob = model.predict_proba(X_test_tfidf)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = auc(fpr, tpr)
        
        # Save model file
        model_filename = name.lower().replace(" ", "_") + "_model.pkl"
        model_save_path = os.path.join(MODELS_DIR, model_filename)
        joblib.dump(model, model_save_path)
        print(f"Saved {name} model to {model_save_path}")
        
        # Update best model info
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            
        # Store in metrics dictionary
        metrics_data[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),  # [ [TN, FP], [FN, TP] ]
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(auc_score),
            "is_transformer": False,
            "is_deep_learning": False
        }
        
    # Set best model metadata
    metrics_data["metadata"] = {
        "best_model_name": best_model_name,
        "best_model_accuracy": float(best_accuracy),
        "dataset_size": len(df),
        "num_real": int((df['label'] == 1).sum()),
        "num_fake": int((df['label'] == 0).sum())
    }
    
    # Save metrics JSON
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_data, f, indent=4)
        
    print(f"Metrics saved to {METRICS_PATH}")
    print("ML Training Complete!")

if __name__ == "__main__":
    main()
