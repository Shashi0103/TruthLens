import os
import json
import pandas as pd
from collections import Counter
import re

# Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
fake_path = os.path.join(base_dir, "dataset/fake.csv")
real_path = os.path.join(base_dir, "dataset/real.csv")
metrics_path = os.path.join(base_dir, "models/metrics.json")

def precompute():
    print("Precomputing visualization data from local CSVs...")
    
    if not os.path.exists(fake_path) or not os.path.exists(real_path):
        print("ERROR: Local CSV files not found!")
        return
        
    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)
    
    # 1. Subject Category Counts
    fake_subjects = fake_df['subject'].value_counts().to_dict()
    real_subjects = real_df['subject'].value_counts().to_dict()
    
    # 2. Top News Sources (Top 10)
    fake_sources = fake_df['source'].value_counts().head(10).to_dict()
    real_sources = real_df['source'].value_counts().head(10).to_dict()
    
    # 3. Top Word Frequencies (Top 15 for charts)
    print("Computing top word frequencies (filtering stopwords)...")
    from nltk.corpus import stopwords
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    
    # Simple tokenizer helper
    def get_top_words(df, num_words=15):
        # Sample 300 articles for representative top words to save processing time
        text_corpus = " ".join(df['text'].fillna('').head(300).values).lower()
        words = re.findall(r'\b[a-z]{4,}\b', text_corpus) # words with at least 4 letters
        filtered_words = [w for w in words if w not in stop_words and w != "said"]
        counter = Counter(filtered_words)
        return counter.most_common(num_words)
        
    fake_words = get_top_words(fake_df)
    real_words = get_top_words(real_df)
    
    # Load current metrics
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = {}
        
    # Append to metadata
    if 'metadata' not in metrics:
        metrics['metadata'] = {}
        
    metrics['metadata']['viz_data'] = {
        "subject_counts": {
            "Fake": fake_subjects,
            "Real": real_subjects
        },
        "source_counts": {
            "Fake": fake_sources,
            "Real": real_sources
        },
        "word_frequencies": {
            "Fake": fake_words,
            "Real": real_words
        }
    }
    
    # Save back
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("SUCCESS: Precomputed visualization data successfully appended to models/metrics.json!")

if __name__ == "__main__":
    precompute()
