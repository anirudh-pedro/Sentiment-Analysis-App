import streamlit as st
import pickle
import numpy as np
import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add explicit NLTK path
nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Import preprocessing - with fallback
try:
    from utils.preprocess import clean_text, extract_features
except ImportError as e:
    st.error(f"Error importing preprocessing utilities: {e}")
    st.error("Please check the utils/preprocess.py file")
    sys.exit(1)

# Error handling for NLTK import
try:
    import nltk
    nltk.data.path.append(nltk_data_dir)  # Add explicit path
    
    # Ensure NLTK data is downloaded
    nltk.download('vader_lexicon', download_dir=nltk_data_dir, quiet=True)
    
    # Import VADER after ensuring resources are available
    from nltk.sentiment import SentimentIntensityAnalyzer
except ImportError as e:
    st.error(f"Error loading NLTK: {e}")
    st.error("Please run 'python fix_nltk.py' to fix NLTK resources")
    st.error("Then run 'python setup_nltk.py' to install required resources")
    sys.exit(1)

try:
    from scipy.sparse import hstack
except ImportError as e:
    st.error(f"Error loading SciPy: {e}")
    st.error("Please install SciPy with: pip install scipy")
    sys.exit(1)

# Load model and components
try:
    with open("model/sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    with open("model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    # Store model features expectation for later validation
    expected_features = getattr(model, 'n_features_in_', None)
    if expected_features:
        print(f"Model expects {expected_features} features")
    
    # Try to load feature columns, but have a fallback if file doesn't exist
    try:
        with open("model/feature_columns.pkl", "rb") as f:
            feature_columns = pickle.load(f)
    except FileNotFoundError:
        print("Warning: feature_columns.pkl not found, using default columns")
        feature_columns = ['vader_compound', 'vader_pos', 'vader_neu', 'vader_neg', 
                          'exclamation_count', 'question_count', 'caps_ratio', 'text_length']
        # Create the file for future use
        with open("model/feature_columns.pkl", "wb") as f:
            pickle.dump(feature_columns, f)
            print("Created feature_columns.pkl with default values")
except Exception as e:
    st.error(f"Error loading model components: {e}")
    st.error("Please ensure the model files exist in the 'model' directory")
    sys.exit(1)

# Initialize VADER sentiment analyzer
try:
    sia = SentimentIntensityAnalyzer()
except Exception as e:
    st.error(f"Error initializing VADER: {e}")
    st.error("Please run setup_nltk.py to download required resources")
    sys.exit(1)

positive_phrases = [
    "not bad", "not terrible", "not awful", "not horrible", 
    "pretty good", "quite good", "very good", "really good",
    "love it", "great", "excellent", "amazing", "fantastic",
    "wonderful", "awesome", "perfect", "brilliant"
]

negative_phrases = [
    "not good", "not great", "not worth", "not recommend",
    "terrible", "awful", "horrible", "worst", "bad", "poor",
    "disappointing", "waste", "useless", "hate", "dislike",
    "unhelpful", "rude", "stopped working", "breaks", "broken"
]

negative_emojis = ["👎", "😡", "😠", "😞", "😟", "😕", "🤬", "😒", "💔", "👿", "😤"]
positive_emojis = ["👍", "😊", "😃", "😄", "😁", "😆", "😍", "🥰", "😘", "❤️", "💕"]

def analyze_sentiment(text):
    """Analyze sentiment with confidence scores and explanations"""
    
    result = {
        "sentiment": None,
        "confidence": 0.0,
        "explanation": [],
        "override": False
    }
    
    if not text.strip():
        return {
            "sentiment": None,
            "confidence": 0.0,
            "explanation": ["No text provided for analysis."],
            "override": False
        }
    
    vader_scores = sia.polarity_scores(text)
    vader_compound = vader_scores['compound']
    
    has_negative_emoji = any(emoji in text for emoji in negative_emojis)
    has_positive_emoji = any(emoji in text for emoji in positive_emojis)
    
    if has_negative_emoji:
        result["sentiment"] = "Negative"
        result["confidence"] = 0.90
        result["explanation"].append(f"Contains negative emoji(s)")
        result["override"] = True
    elif has_positive_emoji:
        result["sentiment"] = "Positive"
        result["confidence"] = 0.90
        result["explanation"].append(f"Contains positive emoji(s)")
        result["override"] = True
    
    lower_text = text.lower()
    
    contains_not_bad = False
    for phrase in positive_phrases:
        if phrase in lower_text:
            if phrase.startswith("not "):
                contains_not_bad = True
                result["sentiment"] = "Positive"
                result["confidence"] = 0.95  # Higher confidence for these clear cases
                result["explanation"].append(f"Contains positive phrase: '{phrase}'")
                result["override"] = True
                break
            else:
                if not result["override"]:  # Don't override emoji sentiment
                    result["sentiment"] = "Positive"
                    result["confidence"] = max(0.85, result["confidence"])
                    result["explanation"].append(f"Contains positive phrase: '{phrase}'")
                    result["override"] = True
    
    if not contains_not_bad:
        for phrase in negative_phrases:
            if phrase in lower_text:
                if result["sentiment"] == "Negative":
                    result["confidence"] = max(0.95, result["confidence"])  # Higher confidence for combined signals
                    result["explanation"].append(f"Contains negative phrase: '{phrase}'")
                elif not result["override"]:
                    result["sentiment"] = "Negative"
                    result["confidence"] = max(0.85, result["confidence"])
                    result["explanation"].append(f"Contains negative phrase: '{phrase}'")
                    result["override"] = True
                
                # Combined emojis and negative words are very strong signals
                if has_negative_emoji:
                    result["confidence"] = 0.98
    
    if vader_compound >= 0.05:
        if not result["override"]:
            result["sentiment"] = "Positive"
        result["confidence"] = max(abs(vader_compound), result["confidence"])
        result["explanation"].append(f"VADER sentiment score: {vader_compound:.2f} (positive)")
    elif vader_compound <= -0.05:
        if not result["override"]:
            result["sentiment"] = "Negative"
        result["confidence"] = max(abs(vader_compound), result["confidence"])
        result["explanation"].append(f"VADER sentiment score: {vader_compound:.2f} (negative)")
    elif vader_compound == 0.0 and not result["override"]:
        if has_negative_emoji or any(phrase in lower_text for phrase in negative_phrases):
            result["sentiment"] = "Negative"
            result["confidence"] = 0.7
            result["explanation"].append("Detected negative signals with neutral VADER score")
    
    if not result["override"] or abs(vader_compound) < 0.5:
        try:
            cleaned_text = clean_text(text)
            features_dict = extract_features(text)
            
            # Create text vector
            text_vector = vectorizer.transform([cleaned_text])
            
            # Create feature vector
            features_array = []
            
            # Add VADER scores
            features_dict['vader_compound'] = vader_scores['compound']
            features_dict['vader_pos'] = vader_scores['pos']
            features_dict['vader_neu'] = vader_scores['neu']
            features_dict['vader_neg'] = vader_scores['neg']
            
            # Ensure all expected features are present
            for feature in feature_columns:
                features_array.append(features_dict.get(feature, 0))
            
            features_vector = np.array([features_array])
            
            # Combine vectors
            combined_vector = hstack([text_vector, features_vector])
            
            # Check feature dimensions - NEW CODE
            expected_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else 5000
            current_features = combined_vector.shape[1]
            
            if current_features != expected_features:
                result["explanation"].append(f"Feature dimension mismatch ({current_features} vs {expected_features}). Using text features only.")
                
                if text_vector.shape[1] > expected_features:
                    text_vector_truncated = text_vector[:, :expected_features]
                    prediction = model.predict(text_vector_truncated)[0]
                else:
                    # Fallback to VADER if we can't fix the dimensions
                    prediction = 1 if vader_compound >= 0 else 0
                    result["explanation"].append("Using VADER as fallback due to feature mismatch.")
            else:
                # Normal path - use combined vector
                prediction = model.predict(combined_vector)[0]
            
            # Get probability if model supports it
            try:
                if hasattr(model, 'predict_proba'):
                    if current_features == expected_features:
                        proba = model.predict_proba(combined_vector)[0]
                    elif text_vector.shape[1] > expected_features:
                        proba = model.predict_proba(text_vector_truncated)[0]
                    else:
                        proba = [0.5, 0.5]  # Fallback
                    confidence = proba[1] if prediction == 1 else proba[0]
                else:
                    confidence = 0.7  # Default
            except:
                confidence = 0.7  # Fallback
            
            ml_sentiment = "Positive" if prediction == 1 else "Negative"
            
            # If ML confidence is higher or no clear sentiment yet
            if confidence > result["confidence"] or not result["sentiment"]:
                result["sentiment"] = ml_sentiment
                result["confidence"] = confidence
                result["explanation"].append(f"ML model prediction: {ml_sentiment} (confidence: {confidence:.2f})")
        
        except Exception as e:
            # Handle any errors during ML prediction
            result["explanation"].append(f"Error during ML prediction: {str(e)}")
            if not result["sentiment"]:
                # Fallback to VADER
                result["sentiment"] = "Positive" if vader_compound >= 0 else "Negative"
                result["confidence"] = abs(vader_compound)
                result["explanation"].append("Using VADER as fallback due to ML error.")
    
    # Final determination
    if not result["sentiment"]:
        result["sentiment"] = "Neutral"
        result["explanation"].append("No clear sentiment detected")
    
    return result

# UI
st.set_page_config(page_title="Advanced Sentiment Analyzer", layout="centered")
st.title("💬 Sentiment Analysis App")
st.markdown("Enter a product review, tweet or any text to analyze its sentiment:")

text = st.text_area("Text Input", height=150)

with st.expander("Advanced Options"):
    show_explanation = st.checkbox("Show analysis explanation", value=True)
    show_confidence = st.checkbox("Show confidence score", value=True)

if st.button("Analyze Sentiment"):
    if text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Get complete sentiment analysis
        analysis = analyze_sentiment(text)
        
        # Display result
        if analysis["sentiment"] == "Positive":
            st.success(f"😊 Positive Sentiment" + (f" ({analysis['confidence']:.2f})" if show_confidence else ""))
        elif analysis["sentiment"] == "Negative":
            st.error(f"😞 Negative Sentiment" + (f" ({analysis['confidence']:.2f})" if show_confidence else ""))
        else:
            st.info(f"😐 Neutral Sentiment" + (f" ({analysis['confidence']:.2f})" if show_confidence else ""))
        
        # Show explanations if enabled
        if show_explanation and analysis["explanation"]:
            st.subheader("Analysis Explanation")
            for explanation in analysis["explanation"]:
                st.markdown(f"- {explanation}")
        
        # Show text processing details in expander
        with st.expander("See text processing details"):
            st.write("**Cleaned text:**")
            st.code(clean_text(text))
            
            st.write("**Extracted features:**")
            features = extract_features(text)
            for k, v in features.items():
                st.markdown(f"- **{k}**: {v}")
            
            st.write("**VADER scores:**")
            vader_scores = sia.polarity_scores(text)
            for k, v in vader_scores.items():
                st.markdown(f"- **{k}**: {v}")
