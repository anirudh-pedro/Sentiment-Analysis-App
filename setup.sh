#!/bin/bash

# Setup script for cloud deployment

# Make NLTK data directory
mkdir -p ~/nltk_data

# Download required NLTK resources
python -m nltk.downloader -d ~/nltk_data punkt
python -m nltk.downloader -d ~/nltk_data stopwords
python -m nltk.downloader -d ~/nltk_data wordnet
python -m nltk.downloader -d ~/nltk_data vader_lexicon
python -m nltk.downloader -d ~/nltk_data averaged_perceptron_tagger
python -m nltk.downloader -d ~/nltk_data omw-1.4

echo "NLTK resources downloaded successfully"

# Create directories if they don't exist
mkdir -p model
mkdir -p data

# Generate feature columns file if it doesn't exist
if [ ! -f "model/feature_columns.pkl" ]; then
    echo "Generating feature columns file..."
    python generate_feature_columns.py
fi

echo "Setup complete!"
