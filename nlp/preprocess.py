import re
import html
import nltk
import spacy
from bs4 import BeautifulSoup

# Programmatically check and download NLTK datasets
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

from nltk.corpus import stopwords

# Global stopwords set
STOPWORDS = set(stopwords.words('english'))

# Load spaCy model with components disabled for speed
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "senter"])
except OSError:
    # Fallback to downloading if not found (though checked previously)
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], stdout=subprocess.DEVNULL)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "senter"])

# Set max length to handle very large articles if needed
nlp.max_length = 2000000

def clean_html(text: str) -> str:
    """Removes HTML tags and unescapes HTML entities."""
    if not text:
        return ""
    # Strip HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Unescape HTML entities (e.g., &amp; -> &)
    text = html.unescape(text)
    return text

def preprocess_text(text: str) -> str:
    """
    Applies the full preprocessing pipeline:
    1. Clean HTML tags
    2. Convert to lowercase
    3. Remove URLs
    4. Remove special characters and punctuation
    5. Tokenize, remove stopwords, and lemmatize using spaCy
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 1. Clean HTML tags
    text = clean_html(text)
    
    # 2. Convert to lowercase
    text = text.lower()
    
    # 3. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 4. Remove special characters and punctuation (keep letters, digits, and spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 5. Lemmatization and stopword removal using spaCy (fast)
    doc = nlp(text)
    cleaned_tokens = []
    for token in doc:
        # Check if the token is a stopword or space or too short
        if token.text not in STOPWORDS and not token.is_space and len(token.text) > 1:
            cleaned_tokens.append(token.lemma_)
            
    return " ".join(cleaned_tokens)

class SimpleTokenizer:
    def __init__(self, max_words=5000):
        self.max_words = max_words
        self.word2idx = {}
        self.idx2word = {}
        
    def fit(self, texts):
        word_counts = {}
        for text in texts:
            for word in text.split():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Special tokens: 0 = PAD, 1 = UNK
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        for i, (word, _) in enumerate(sorted_words[:self.max_words - 2]):
            self.word2idx[word] = i + 2
            
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
    def text_to_sequence(self, text, max_len=100):
        words = text.split()
        seq = []
        for w in words[:max_len]:
            seq.append(self.word2idx.get(w, 1)) # Default to <UNK>
        # Pad sequence
        if len(seq) < max_len:
            seq += [0] * (max_len - len(seq))
        return seq

if __name__ == "__main__":
    test_text = "<html><body>Verify this <a href='https://example.com'>news article</a>! It's running on CPU.</body></html>"
    print("Original:", test_text)
    print("Cleaned: ", preprocess_text(test_text))

