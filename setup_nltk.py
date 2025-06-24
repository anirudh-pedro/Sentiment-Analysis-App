"""
NLTK Resource Downloader - Ensures all required NLTK resources are downloaded
"""
import os
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

def ensure_nltk_resources():
    """Ensure all required NLTK resources are downloaded"""
    print("Checking and downloading required NLTK resources...")
    
    # Create a directory for NLTK data
    nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
        print(f"Created NLTK data directory: {nltk_data_dir}")
    
    # List of required NLTK resources
    required_resources = [
        'punkt',
        'stopwords',
        'wordnet',
        'vader_lexicon',
        'averaged_perceptron_tagger',
        'omw-1.4'
    ]
    
    # Download each resource
    for resource in required_resources:
        try:
            print(f"Downloading '{resource}'...")
            nltk.download(resource, download_dir=nltk_data_dir)
            print(f"✓ Successfully downloaded '{resource}'")
        except Exception as e:
            print(f"❌ Error downloading '{resource}': {e}")
    
    # Verify resources by using them
    try:
        # Test stopwords
        print("Testing stopwords...")
        stop_words = stopwords.words('english')
        print(f"✓ Stopwords loaded successfully ({len(stop_words)} stopwords)")
        
        # Test wordnet
        print("Testing wordnet...")
        lemmatizer = WordNetLemmatizer()
        test_lemma = lemmatizer.lemmatize('running')
        print(f"✓ WordNet lemmatizer working: 'running' -> '{test_lemma}'")
        
        # Test punkt
        print("Testing punkt tokenizer...")
        tokens = word_tokenize("This is a test sentence.")
        print(f"✓ Punkt tokenizer working: tokenized into {len(tokens)} tokens")
        
        # Test vader
        print("Testing VADER sentiment analyzer...")
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores("This is great!")
        print(f"✓ VADER working: sentiment of 'This is great!' -> {sentiment}")
        
    except Exception as e:
        print(f"❌ Error testing NLTK resources: {e}")
        print("Please try running this script again or manually install the missing resources.")
    
    print("\nNLTK setup complete!")

if __name__ == "__main__":
    ensure_nltk_resources()
