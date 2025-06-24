"""
Sentiment Analysis Validation Script - Tests the app's sentiment analysis on various edge cases
"""
import sys
import warnings
import os
import pickle
import nltk
import pandas as pd
from utils.preprocess import clean_text, extract_features
from nltk.sentiment import SentimentIntensityAnalyzer

# Suppress warnings
warnings.filterwarnings('ignore')

# Load app.py's analyze_sentiment function
sys.path.append('.')
try:
    from app import analyze_sentiment, sia
    print("Successfully imported analyze_sentiment function")
except ImportError as e:
    print(f"ERROR importing analyze_sentiment: {e}")
    sys.exit(1)

# Test cases - intentionally challenging
test_cases = [
    # Positive cases
    "This product is amazing! I absolutely love it.",
    "Not bad at all, actually quite good.",
    "This movie is not terrible, I actually enjoyed it.",
    "Despite a few minor issues, I'm very satisfied with my purchase 😊",
    
    # Negative cases  
    "The customer service was unhelpful 👎",
    "I really hated the product quality.",
    "The app keeps crashing and losing my data 😡",
    "This was the worst experience I've ever had with any company.",
    
    # Mixed/neutral cases
    "The first half of the movie was boring, but the ending was amazing!",
    "It has good features but also some issues.",
    "I'm not sure if I like it or not.",
    
    # Edge cases
    "Not good",
    "Not bad",
    "👍",
    "👎",
    "The food was ok."
]

# Run tests
print("\n" + "="*80)
print("SENTIMENT ANALYSIS VALIDATION TEST")
print("="*80)

for i, text in enumerate(test_cases):
    print(f"\n[TEST CASE {i+1}] '{text}'")
    
    # Get analysis
    analysis = analyze_sentiment(text)
    
    # Show results
    print(f"Result: {analysis['sentiment']} (Confidence: {analysis['confidence']:.2f})")
    
    # Show explanations
    print("Explanation:")
    for explanation in analysis['explanation']:
        print(f"  - {explanation}")
    
    # Clean text and features
    print(f"Cleaned text: '{clean_text(text)}'")
    
    # VADER scores
    vader_scores = sia.polarity_scores(text)
    print(f"VADER scores: {vader_scores}")
    
    # Separator
    print("-"*80)

print("\nValidation complete. Please review the results above.")
