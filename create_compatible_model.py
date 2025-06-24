"""
Simplified Model Retraining - Creates a compatible model for the current feature extraction
"""
import pickle
import os
import sys
import numpy as np
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.preprocess import clean_text, extract_features
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

def create_compatible_model():
    """Create a simplified model compatible with current features"""
    print("Creating a simplified compatible model...")
    
    # Ensure NLTK resources
    nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
    nltk.download('vader_lexicon', download_dir=nltk_data_dir, quiet=True)
    
    # Simple positive and negative samples for training
    samples = [
        "This is excellent!", 
        "I love this product", 
        "Amazing service", 
        "Great experience",
        "Very happy with my purchase",
        "This is fantastic",
        "Wonderful product, highly recommend",
        "Best purchase I've ever made",
        "Exceeded my expectations",
        "This is terrible",
        "I hate this product",
        "Poor service",
        "Bad experience",
        "Very disappointed",
        "This is awful",
        "Worst purchase ever",
        "Completely useless",
        "Don't recommend at all"
    ]
    
    # Labels (1 for positive, 0 for negative)
    labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Create dataframe
    df = pd.DataFrame({'text': samples, 'label': labels})
    
    # Clean text
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Extract features
    features_list = []
    for text in df['text']:
        features_list.append(extract_features(text))
    
    # Convert to dataframe
    features_df = pd.DataFrame(features_list)
    
    # Get feature columns
    feature_columns = features_df.columns.tolist()
    print(f"Feature columns: {feature_columns}")
    
    # Save feature columns
    with open("model/feature_columns.pkl", "wb") as f:
        pickle.dump(feature_columns, f)
        print("Saved feature_columns.pkl")
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000-len(feature_columns))
    text_vectors = vectorizer.fit_transform(df['clean_text'])
    
    # Add VADER scores
    sia = SentimentIntensityAnalyzer()
    for i, text in enumerate(df['text']):
        sentiment = sia.polarity_scores(text)
        features_df.loc[i, 'vader_compound'] = sentiment['compound']
        features_df.loc[i, 'vader_pos'] = sentiment['pos']
        features_df.loc[i, 'vader_neu'] = sentiment['neu']
        features_df.loc[i, 'vader_neg'] = sentiment['neg']
    
    # Create combined feature vector
    combined_features = hstack([text_vectors, features_df.values])
    print(f"Total features: {combined_features.shape[1]}")
    
    # Train a simple logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(combined_features, df['label'])
    
    # Save model and vectorizer
    with open("model/sentiment_model.pkl", "wb") as f:
        pickle.dump(model, f)
        print("Saved sentiment_model.pkl")
    
    with open("model/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
        print("Saved vectorizer.pkl")
    
    print("\n✅ Created compatible model with current feature extraction")
    print("Expected features:", combined_features.shape[1])
    
    return True

if __name__ == "__main__":
    create_compatible_model()
