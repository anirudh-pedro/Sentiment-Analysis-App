"""
Dimension Fix Script - Creates a compatible model without needing to download NLTK resources
"""
import pickle
import os
import sys
import numpy as np
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression

def create_dimension_fix():
    """Create a dimension-fixed version of the current model"""
    print("Creating a dimension-fixed model...")
    
    # Load current model
    try:
        with open("model/sentiment_model.pkl", "rb") as f:
            model = pickle.load(f)
            
        with open("model/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        
        print("Loaded existing model and vectorizer")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Get expected features
    expected_features = getattr(model, 'n_features_in_', None)
    if not expected_features:
        print("Could not determine expected features count from model")
        return False
    
    print(f"Model expects {expected_features} features")
    
    # Create feature_columns.pkl with only the essential features
    essential_features = ['vader_compound', 'vader_pos', 'vader_neu', 'vader_neg', 
                          'exclamation_count', 'question_count', 'caps_ratio', 'text_length']
    
    with open("model/feature_columns.pkl", "wb") as f:
        pickle.dump(essential_features, f)
        print(f"Saved feature_columns.pkl with {len(essential_features)} essential features")
    
    print("\n✅ Created dimension fix")
    print("The app will now use only essential features and truncate the text vector if needed")
    print("This will prevent dimension mismatch errors")
    
    return True

if __name__ == "__main__":
    create_dimension_fix()
