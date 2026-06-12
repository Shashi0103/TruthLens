import os
import sys
import argparse
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification

# Add project root to python path to import nlp/preprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nlp.preprocess import clean_html

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
BERT_SAVE_DIR = os.path.join(MODELS_DIR, "bert_model")
DISTILBERT_SAVE_DIR = os.path.join(MODELS_DIR, "distilbert_model")

# Use a tiny BERT to train instantly on CPU
BERT_MODEL_NAME = "prajjwal1/bert-tiny" # 17MB
# For DistilBERT, use the standard distilbert-base-uncased model
DISTILBERT_MODEL_NAME = "distilbert-base-uncased" # 260MB

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_data(quick_mode=True):
    print("Loading data from CSVs...")
    fake_df = pd.read_csv(os.path.join(BASE_DIR, "dataset/fake.csv"))
    real_df = pd.read_csv(os.path.join(BASE_DIR, "dataset/real.csv"))
    
    fake_df['label'] = 0
    real_df['label'] = 1
    
    if quick_mode:
        print("Quick mode enabled for Transformers: Subsampling 30 fake and 30 real articles...")
        fake_df = fake_df.sample(min(30, len(fake_df)), random_state=42)
        real_df = real_df.sample(min(30, len(real_df)), random_state=42)
        
    df = pd.concat([fake_df, real_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def train_transformer(model_name, save_dir, train_loader, val_loader, epochs=1):
    print(f"Loading pre-trained model: {model_name}...")
    if "distilbert" in model_name.lower():
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    print(f"Fine-tuning {model_name} on CPU...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} Complete. Average Loss: {total_loss/len(train_loader):.4f}")
        
    # Save the fine-tuned model and tokenizer
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved fine-tuned model and tokenizer to {save_dir}")
    
    # Evaluate
    model.eval()
    y_true = []
    y_probs = []
    y_pred = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            # Softmax to get probabilities for the positive class (label 1)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
            y_true.extend(labels.tolist())
            y_probs.extend(probs.tolist())
            y_pred.extend(preds.tolist())
            
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = np.array(y_pred)
    
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = auc(fpr, tpr)
    
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": float(auc_score),
        "is_transformer": True,
        "is_deep_learning": False
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", default=True, help="Train on a small subset")
    parser.add_argument("--full", action="store_false", dest="quick", help="Train on the full dataset")
    args = parser.parse_args()
    
    # 1. Load and clean
    df = load_data(quick_mode=args.quick)
    # Just clean HTML tags for transformer input (transformers do their own subword tokenization, lowercase and URL cleaning)
    df['clean_text'] = df['title'].fillna('') + " " + df['text'].fillna('').apply(clean_html)
    df = df[df['clean_text'].str.strip() != ''].reset_index(drop=True)
    
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    X_train, X_val, y_train, y_val = train_test_split(
        df['clean_text'].values, df['label'].values, test_size=0.2, random_state=42, stratify=df['label']
    )
    
    train_dataset = NewsDataset(X_train, y_train, bert_tokenizer)
    val_dataset = NewsDataset(X_val, y_val, bert_tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # 3. Train BERT
    bert_eval = train_transformer(
        BERT_MODEL_NAME, BERT_SAVE_DIR, train_loader, val_loader, epochs=1
    )
    
    # 4. Setup DataLoader and Train DistilBERT
    print("\nPreparing DistilBERT...")
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_MODEL_NAME)
    
    train_dataset_db = NewsDataset(X_train, y_train, distilbert_tokenizer)
    val_dataset_db = NewsDataset(X_val, y_val, distilbert_tokenizer)
    
    train_loader_db = DataLoader(train_dataset_db, batch_size=8, shuffle=True)
    val_loader_db = DataLoader(val_dataset_db, batch_size=8, shuffle=False)
    
    distilbert_eval = train_transformer(
        DISTILBERT_MODEL_NAME, DISTILBERT_SAVE_DIR, train_loader_db, val_loader_db, epochs=1
    )
    
    # 5. Append Transformer metrics to metrics.json
    metrics_data = {}
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                metrics_data = json.load(f)
        except Exception:
            pass
            
    metrics_data["BERT"] = bert_eval
    metrics_data["DistilBERT"] = distilbert_eval
    
    # Check if this changes the best model
    metadata = metrics_data.get("metadata", {})
    best_acc = metadata.get("best_model_accuracy", 0.0)
    best_name = metadata.get("best_model_name", "")
    
    if bert_eval["accuracy"] > best_acc:
        best_acc = bert_eval["accuracy"]
        best_name = "BERT"
    if distilbert_eval["accuracy"] > best_acc:
        best_acc = distilbert_eval["accuracy"]
        best_name = "DistilBERT"
        
    metadata["best_model_name"] = best_name
    metadata["best_model_accuracy"] = best_acc
    metrics_data["metadata"] = metadata
    
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_data, f, indent=4)
        
    print(f"\nTransformer metrics appended to {METRICS_PATH}")
    print("BERT/DistilBERT Training Complete!")

if __name__ == "__main__":
    main()
