"""
Model Compatibility Check - Verifies that the trained model is compatible with the current feature extraction
"""
import pickle
import os
import sys
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

def check_model_compatibility():
    """Check if the model is compatible with the current feature extraction"""
    print("Checking model compatibility...")
    
    # Load model and vectorizer
    try:
        with open("model/sentiment_model.pkl", "rb") as f:
            model = pickle.load(f)
        
        with open("model/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        
        # Try to load feature columns
        try:
            with open("model/feature_columns.pkl", "rb") as f:
                feature_columns = pickle.load(f)
        except FileNotFoundError:
            feature_columns = ['vader_compound', 'vader_pos', 'vader_neu', 'vader_neg', 
                              'exclamation_count', 'question_count', 'caps_ratio', 'text_length']
        
        print(f"Loaded model and vectorizer successfully")
        print(f"Feature columns: {feature_columns}")
    except Exception as e:
        print(f"Error loading model components: {e}")
        return False
    
    # Get expected number of features
    expected_features = getattr(model, 'n_features_in_', None)
    if not expected_features:
        print("Warning: Could not determine expected feature count from model")
        return False
    
    print(f"Model expects {expected_features} features")
    
    # Test with a sample text
    sample_text = "This is a test sample to check model compatibility"
    
    # Generate vector
    text_vector = vectorizer.transform([sample_text])
    
    # Generate dummy additional features
    dummy_features = np.zeros((1, len(feature_columns)))
    
    # Combine vectors
    combined_vector = hstack([text_vector, dummy_features])
    
    print(f"Generated vector with {combined_vector.shape[1]} features")
    
    # Check if dimensions match
    if combined_vector.shape[1] != expected_features:
        print(f"❌ DIMENSION MISMATCH: Model expects {expected_features} features, but we're generating {combined_vector.shape[1]} features")
        
        # Calculate difference
        diff = combined_vector.shape[1] - expected_features
        print(f"Difference: {diff} features")
        
        if diff > 0:
            print("The current feature extraction is generating MORE features than the model expects.")
            print("Possible solutions:")
            print("1. Retrain the model with the current feature set")
            print("2. Modify the app.py to truncate features (already implemented)")
        else:
            print("The current feature extraction is generating FEWER features than the model expects.")
            print("Possible solutions:")
            print("1. Retrain the model with the current feature set")
            print("2. Add dummy features to match the expected count")
        
        return False
    else:
        print(f"✅ DIMENSIONS MATCH: Model expects {expected_features} features and we're generating {combined_vector.shape[1]} features")
        return True

if __name__ == "__main__":
    if check_model_compatibility():
        print("\n✅ Model is compatible with current feature extraction")
        sys.exit(0)
    else:
        print("\n❌ Model is NOT compatible with current feature extraction")
        print("Run the app anyway - it includes fallback mechanisms for handling mismatches")
        sys.exit(1)
