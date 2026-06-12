import os
import sys
import argparse
import joblib
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc

# Add project root to python path to import nlp/preprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nlp.preprocess import preprocess_text, SimpleTokenizer

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
VOCAB_PATH = os.path.join(MODELS_DIR, "lstm_vocab.pkl")
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model.pt")
BILSTM_MODEL_PATH = os.path.join(MODELS_DIR, "bilstm_model.pt")

# Hyperparameters
MAX_LEN = 100
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001


# PyTorch LSTM Classifier
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=1, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            batch_first=True, 
            bidirectional=bidirectional
        )
        self.bidirectional = bidirectional
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, text):
        embedded = self.embedding(text)
        # lstm_out: [batch_size, seq_len, hidden_dim * num_directions]
        # hidden: [num_layers * num_directions, batch_size, hidden_dim]
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Concat final forward and backward hidden states if bidirectional
        if self.bidirectional:
            hidden_last = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden_last = hidden[-1,:,:]
            
        dense_out = self.fc(hidden_last)
        return self.sigmoid(dense_out)

def load_data(quick_mode=True):
    print("Loading data from CSVs...")
    fake_df = pd.read_csv(os.path.join(BASE_DIR, "dataset/fake.csv"))
    real_df = pd.read_csv(os.path.join(BASE_DIR, "dataset/real.csv"))
    
    fake_df['label'] = 0
    real_df['label'] = 1
    
    if quick_mode:
        print("Quick mode enabled for LSTM: Subsampling 200 fake and 200 real articles...")
        fake_df = fake_df.sample(min(200, len(fake_df)), random_state=42)
        real_df = real_df.sample(min(200, len(real_df)), random_state=42)
        
    df = pd.concat([fake_df, real_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        total_train = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_x).squeeze(1)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            preds_bin = (predictions >= 0.5).float()
            train_correct += (preds_bin == batch_y).sum().item()
            total_train += batch_x.size(0)
            
        train_loss /= total_train
        train_acc = train_correct / total_train
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions = model(batch_x).squeeze(1)
                loss = criterion(predictions, batch_y)
                
                val_loss += loss.item() * batch_x.size(0)
                preds_bin = (predictions >= 0.5).float()
                val_correct += (preds_bin == batch_y).sum().item()
                total_val += batch_x.size(0)
                
        val_loss /= total_val
        val_acc = val_correct / total_val
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
    return history

def evaluate_model(model, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    y_true = []
    y_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            predictions = model(batch_x).squeeze(1)
            y_true.extend(batch_y.tolist())
            y_probs.extend(predictions.tolist())
            
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= 0.5).astype(int)
    
    # Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC Curve
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
        "is_transformer": False,
        "is_deep_learning": True
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", default=True, help="Train on a small subset")
    parser.add_argument("--full", action="store_false", dest="quick", help="Train on the full dataset")
    args = parser.parse_args()
    
    # 1. Load and Clean
    df = load_data(quick_mode=args.quick)
    df['combined_text'] = df['title'].fillna('') + " " + df['text'].fillna('')
    df['clean_text'] = df['combined_text'].apply(preprocess_text)
    df = df[df['clean_text'].str.strip() != ''].reset_index(drop=True)
    
    # 2. Tokenize and Build Vocab
    print("Fitting Tokenizer and Prepping Sequences...")
    tokenizer = SimpleTokenizer(max_words=10000)
    tokenizer.fit(df['clean_text'])
    
    # Save vocabulary dictionary for Streamlit predictions
    joblib.dump(tokenizer, VOCAB_PATH)
    print(f"Saved vocabulary to {VOCAB_PATH}")
    
    # Convert text to padded index sequences
    sequences = [tokenizer.text_to_sequence(t, max_len=MAX_LEN) for t in df['clean_text']]
    X = torch.tensor(sequences, dtype=torch.long)
    y = torch.tensor(df['label'].values, dtype=torch.float)
    
    # Train/Val Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    vocab_size = len(tokenizer.word2idx)
    criterion = nn.BCELoss()
    
    # 3. Train Standard LSTM
    print("\nTraining Standard LSTM model...")
    lstm_model = LSTMClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, bidirectional=False)
    optimizer = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
    lstm_history = train_model(lstm_model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS)
    lstm_eval = evaluate_model(lstm_model, val_loader)
    lstm_eval["history"] = lstm_history
    
    # Save LSTM model
    torch.save(lstm_model.state_dict(), LSTM_MODEL_PATH)
    print(f"Saved LSTM Model to {LSTM_MODEL_PATH}")
    
    # 4. Train Bi-LSTM
    print("\nTraining Bi-LSTM model...")
    bilstm_model = LSTMClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, bidirectional=True)
    optimizer = optim.Adam(bilstm_model.parameters(), lr=LEARNING_RATE)
    bilstm_history = train_model(bilstm_model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS)
    bilstm_eval = evaluate_model(bilstm_model, val_loader)
    bilstm_eval["history"] = bilstm_history
    
    # Save Bi-LSTM model
    torch.save(bilstm_model.state_dict(), BILSTM_MODEL_PATH)
    print(f"Saved Bi-LSTM Model to {BILSTM_MODEL_PATH}")
    
    # 5. Load existing metrics.json and append LSTM metrics
    metrics_data = {}
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                metrics_data = json.load(f)
        except Exception:
            pass
            
    metrics_data["LSTM"] = lstm_eval
    metrics_data["Bi-LSTM"] = bilstm_eval
    
    # Check if this changes the best model
    metadata = metrics_data.get("metadata", {})
    best_acc = metadata.get("best_model_accuracy", 0.0)
    best_name = metadata.get("best_model_name", "")
    
    if lstm_eval["accuracy"] > best_acc:
        best_acc = lstm_eval["accuracy"]
        best_name = "LSTM"
    if bilstm_eval["accuracy"] > best_acc:
        best_acc = bilstm_eval["accuracy"]
        best_name = "Bi-LSTM"
        
    metadata["best_model_name"] = best_name
    metadata["best_model_accuracy"] = best_acc
    metrics_data["metadata"] = metadata
    
    # Write back to metrics.json
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_data, f, indent=4)
        
    print(f"Deep learning metrics appended to {METRICS_PATH}")
    print("LSTM/Bi-LSTM Training Complete!")

if __name__ == "__main__":
    main()
