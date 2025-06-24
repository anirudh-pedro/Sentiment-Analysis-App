import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import os

# Create a directory for NLTK data if it doesn't exist
nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Download ALL required NLTK data with explicit paths
nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
nltk.download('vader_lexicon', download_dir=nltk_data_dir, quiet=True)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir, quiet=True)
nltk.download('omw-1.4', download_dir=nltk_data_dir, quiet=True)

# Safe resource checking without using nltk.data.find
resource_mapping = {
    'stopwords': 'corpora/stopwords',
    'wordnet': 'corpora/wordnet',
    'punkt': 'tokenizers/punkt',
    'vader_lexicon': 'sentiment/vader_lexicon'
}

for resource, path in resource_mapping.items():
    try:
        # Try accessing the resource to see if it's available
        if resource == 'stopwords':
            stopwords.words('english')
        elif resource == 'wordnet':
            lemmatizer = WordNetLemmatizer()
            lemmatizer.lemmatize('test')
        elif resource == 'punkt':
            word_tokenize('This is a test.')
        # Skip checking vader_lexicon as it will be verified later
    except LookupError:
        print(f"WARNING: Resource {resource} not found, attempting to download again...")
        nltk.download(resource, download_dir=nltk_data_dir, quiet=False)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Enhanced negation words and patterns
NEGATION_WORDS = {'not', 'no', 'never', 'nothing', 'nowhere', 'noone', 'none', 'neither', 'nor', 'cannot', "can't", "won't", "shouldn't", "wouldn't", "couldn't", "doesn't", "don't", "didn't", "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't"}

# Emoji sentiment dictionary (common emojis and their sentiment)
EMOJI_SENTIMENT = {
    '😊': 1, '😃': 1, '😄': 1, '😁': 1, '😆': 1, '😍': 1, '🥰': 1, '😘': 1, '👍': 1, '❤️': 1, '💕': 1,
    '😢': 0, '😭': 0, '😞': 0, '😔': 0, '😟': 0, '😕': 0, '😣': 0, '😖': 0, '😫': 0, '😩': 0, '👎': 0, '💔': 0
}

def clean_text(text):
    if not text or pd.isna(text):
        return ""
    
    # Extract emoji sentiment before lowercasing
    emoji_score = 0
    emoji_count = 0
    for emoji, sentiment in EMOJI_SENTIMENT.items():
        count = text.count(emoji)
        if count > 0:
            emoji_count += count
            emoji_score += sentiment * count
    
    # Store emoji sentiment ratio for later use
    emoji_sentiment_ratio = emoji_score / emoji_count if emoji_count > 0 else 0.5
    
    text = str(text).lower()
    
    # Handle contractions more comprehensively
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'ve": " have", "'ll": " will",
        "'d": " would", "'m": " am", "'s": " is"
    }
    
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Remove URLs, mentions, hashtags
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Handle repeated characters (e.g., "sooooo" -> "so")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
      # Special handling for common expressions - enhance positive conversion
    text = re.sub(r'\bnot bad\b', 'good positive', text)
    text = re.sub(r'\bnot terrible\b', 'good positive', text)
    text = re.sub(r'\bnot awful\b', 'good positive', text)
    text = re.sub(r'\bnot horrible\b', 'good positive', text)
    text = re.sub(r'\bnot (.+) at all\b', r'very good \1', text)  # "not bad at all" → "very good bad"
    
    # Enhanced negation handling with a wider context window
    # Look for negation words followed by up to 5 words
    text = re.sub(
        r'\b(not|no|never|nothing|nowhere|noone|none|neither|nor|cannot|will not)\s+(\w+)(?:\s+(\w+))?(?:\s+(\w+))?(?:\s+(\w+))?(?:\s+(\w+))?', 
        lambda m: f"not_{m.group(2)}" + 
                 (f"_not_{m.group(3)}" if m.group(3) else "") + 
                 (f"_not_{m.group(4)}" if m.group(4) else "") +
                 (f"_not_{m.group(5)}" if m.group(5) else "") +
                 (f"_not_{m.group(6)}" if m.group(6) else ""), 
        text
    )
    
    # Remove punctuation but keep negation markers
    text = re.sub(r"[^\w\s_]", ' ', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and lemmatize with error handling
    try:
        tokens = word_tokenize(text)
    except LookupError:
        # Fallback if NLTK tokenization fails
        print("NLTK tokenization failed, using simple tokenization")
        tokens = text.split()
    
    # Try to use lemmatizer with error handling
    try:
        # Remove stopwords but keep negation-modified words
        tokens = [lemmatizer.lemmatize(word) for word in tokens 
                if word not in stop_words or word.startswith("not_") or len(word) <= 2]
    except LookupError:
        # Fallback if lemmatization fails
        print("NLTK lemmatization failed, skipping lemmatization")
        tokens = [word for word in tokens 
                if word not in stop_words or word.startswith("not_") or len(word) <= 2]
    
    # Remove very short words except important ones
    tokens = [word for word in tokens if len(word) > 2 or word in ['ok', 'no', 'go', 'up']]
    
    # Add emoji sentiment as a special token if emojis were present
    if emoji_count > 0:
        if emoji_sentiment_ratio > 0.5:
            tokens.append('positive_emoji_sentiment')
        elif emoji_sentiment_ratio < 0.5:
            tokens.append('negative_emoji_sentiment')
    
    return ' '.join(tokens)

def extract_features(text):
    """Extract additional features from text"""
    features = {}
    
    # Count exclamation marks and question marks
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    
    # Count capital letters (enthusiasm indicator)
    features['caps_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    
    # Text length
    features['text_length'] = len(text.split())
    
    # Count positive and negative emojis with error handling
    try:
        positive_emoji_count = sum(text.count(emoji) for emoji, sentiment in EMOJI_SENTIMENT.items() if sentiment == 1)
        negative_emoji_count = sum(text.count(emoji) for emoji, sentiment in EMOJI_SENTIMENT.items() if sentiment == 0)
    except Exception as e:
        # Fallback if emoji counting fails
        print(f"Error counting emojis: {e}")
        positive_emoji_count = 0
        negative_emoji_count = 0
    
    features['positive_emoji_count'] = positive_emoji_count
    features['negative_emoji_count'] = negative_emoji_count
    
    # Safely calculate emoji sentiment ratio
    total_emojis = positive_emoji_count + negative_emoji_count
    if total_emojis > 0:
        features['emoji_sentiment_ratio'] = positive_emoji_count / total_emojis
    else:
        features['emoji_sentiment_ratio'] = 0.5  # Neutral default
    
    # Count words with negation
    try:
        features['negation_count'] = len(re.findall(r'\bnot_\w+', text))
    except:
        features['negation_count'] = 0
    
    return features
