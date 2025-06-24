import pickle
import os
import nltk
import sys
import numpy as np
from scipy.sparse import hstack
from nltk.sentiment import SentimentIntensityAnalyzer
sys.path.append('.')
from utils.preprocess import clean_text, extract_features

print("Loading model components...")
# Load model components
with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

try:
    with open('model/feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    print(f"Feature columns: {feature_columns}")
except FileNotFoundError:
    feature_columns = ['vader_compound', 'vader_pos', 'vader_neu', 'vader_neg', 
                     'exclamation_count', 'question_count', 'caps_ratio', 'text_length']
    print(f"Using default feature columns: {feature_columns}")

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Test with samples
samples = [
    'This movie is not bad at all, I actually enjoyed it quite a bit.',
    'The customer service was unhelpful and rude 👎',
    'The customer service was unhelpful 👎',
    'I really hated the product quality',
    'The product stopped working after just two days.'
]

for text in samples:
    print(f"\nAnalyzing: '{text}'")
    # Clean text
    cleaned = clean_text(text)
    print(f"Cleaned text: '{cleaned}'")
    
    # Extract features
    features = extract_features(text)
    print(f"Features: {features}")
    
    # VADER analysis
    vader_scores = sia.polarity_scores(text)
    print(f"VADER scores: {vader_scores}")
    
    # Check for positive/negative patterns
    positive_patterns = ["not bad", "not terrible", "enjoy", "good", "great"]
    negative_patterns = ["unhelpful", "rude", "bad", "terrible", "hate", "stopped working"]
    negative_emojis = ["👎", "😡", "😠", "😞", "😟", "😕", "🤬", "😒"]
    
    positive_matches = [p for p in positive_patterns if p in text.lower()]
    negative_matches = [p for p in negative_patterns if p in text.lower()]
    emoji_matches = [e for e in negative_emojis if e in text]
    
    print(f"Positive patterns found: {positive_matches}")
    print(f"Negative patterns found: {negative_matches}")
    print(f"Negative emojis found: {emoji_matches}")
    
    # Predicted sentiment based on rule-based logic
    if "not bad" in text.lower() or "not terrible" in text.lower():
        predicted = "POSITIVE (rule-based)"
    # Prioritize negative words with emojis
    elif (any(p in text.lower() for p in negative_patterns) and any(em in text for em in negative_emojis)):
        predicted = "NEGATIVE (pattern+emoji)"
    # Check negative emojis alone
    elif any(em in text for em in negative_emojis):
        predicted = "NEGATIVE (emoji-based)"
    # Check negative words alone
    elif any(p in text.lower() for p in negative_patterns):
        predicted = "NEGATIVE (pattern-based)"
    elif vader_scores["compound"] >= 0.05:
        predicted = "POSITIVE (VADER)"
    elif vader_scores["compound"] <= -0.05:
        predicted = "NEGATIVE (VADER)"
    else:
        predicted = "NEUTRAL (VADER)"
    
    print(f"Rule-based prediction: {predicted}")
