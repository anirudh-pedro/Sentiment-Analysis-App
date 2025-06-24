"""
Safe Starter Script - Runs the sentiment analyzer with error handling
"""
import os
import sys
import subprocess
import time

def safe_start():
    """Start the sentiment analyzer safely"""
    print("🚀 Starting Sentiment Analyzer Safely...")
    
    # Check if NLTK resources exist
    nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
    required_dirs = [
        os.path.join(nltk_data_dir, 'tokenizers', 'punkt'),
        os.path.join(nltk_data_dir, 'corpora', 'stopwords'),
        os.path.join(nltk_data_dir, 'sentiment', 'vader_lexicon')
    ]
    
    missing_resources = [d for d in required_dirs if not os.path.exists(d)]
    
    if missing_resources:
        print("⚠️ Some NLTK resources are missing. Running fix script...")
        try:
            subprocess.run([sys.executable, "fix_nltk.py"], check=True)
            print("✅ NLTK resources fixed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error fixing NLTK resources: {e}")
            print("Please manually run: python fix_nltk.py")
            return False
    
    # Run the app directly with streamlit
    print("\n🚀 Starting Streamlit app...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    
    return True

if __name__ == "__main__":
    safe_start()
