"""
NLTK Fix Script - Fixes NLTK resource loading issues by ensuring all resources are properly downloaded
and accessible.
"""
import os
import sys
import nltk
import shutil
import importlib

def fix_nltk_resources():
    """Fix NLTK resource loading issues"""
    print("Starting NLTK resource fix...")
    
    # Get all possible NLTK data directories
    nltk_data_dirs = nltk.data.path
    print(f"Current NLTK data paths: {nltk_data_dirs}")
    
    # Create a primary NLTK data directory in user's home
    primary_nltk_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
    if not os.path.exists(primary_nltk_dir):
        os.makedirs(primary_nltk_dir)
        print(f"Created primary NLTK data directory: {primary_nltk_dir}")
    
    # Ensure primary directory is in path
    if primary_nltk_dir not in nltk_data_dirs:
        nltk.data.path.append(primary_nltk_dir)
        print(f"Added {primary_nltk_dir} to NLTK data path")
    
    # Resources to download with their specific paths
    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'vader_lexicon': 'sentiment/vader_lexicon',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
        'omw-1.4': 'corpora/omw-1.4'
    }
    
    # Force download all resources
    print("\nForcing download of all required resources...")
    for resource, path in resources.items():
        try:
            print(f"Downloading {resource}...")
            nltk.download(resource, download_dir=primary_nltk_dir, quiet=False, force=True)
        except Exception as e:
            print(f"Error downloading {resource}: {e}")
    
    # Reload NLTK modules to ensure they use the new resources
    print("\nReloading NLTK modules...")
    try:
        importlib.reload(nltk)
        print("Successfully reloaded NLTK")
    except Exception as e:
        print(f"Error reloading NLTK: {e}")
    
    print("\nNLTK fix complete! Please restart your application.")

if __name__ == "__main__":
    fix_nltk_resources()
