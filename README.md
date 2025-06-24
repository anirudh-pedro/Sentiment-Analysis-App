# Advanced Sentiment Analyzer

A machine learning-based sentiment analysis application that predicts whether a given text (tweet, review, comment) is positive or negative with high accuracy.

## Features

- **Accurate Sentiment Prediction**: Uses a combination of ML models, rule-based patterns, and lexicon-based techniques
- **Detailed Analysis**: Provides confidence scores and explanations for predictions
- **Advanced Text Processing**: Handles negation, emojis, slang, and contractions
- **Interactive UI**: Easy-to-use Streamlit interface with advanced options

## Project Structure

```
sentiment-analyzer/
├── app.py                       # Streamlit web application
├── train_model.py               # Model training script
├── requirements.txt             # Project dependencies
├── setup_nltk.py                # NLTK resources downloader
├── fix_nltk.py                  # NLTK resource troubleshooter
├── check_model_compatibility.py # Model feature compatibility checker
├── fix_dimensions.py            # Fix feature dimension issues
├── create_compatible_model.py   # Create model with compatible dimensions
├── generate_feature_columns.py  # Generate feature columns file
├── validate_sentiment.py        # Validation test script
├── test_sentiment.py            # Quick test script
├── safe_start.py                # Safe starter with error handling
├── start_app.py                 # App starter with setup
├── run.py                       # Full setup and run script
├── DEPLOYMENT.md                # Deployment checklist and guide
├── data/                        # Dataset directory
│   └── sentiment140.csv         # Training dataset
├── model/                       # Saved model files
│   ├── sentiment_model.pkl
│   ├── vectorizer.pkl
│   └── feature_columns.pkl
└── utils/                       # Utility functions
    └── preprocess.py            # Text preprocessing functions with advanced features
```

## Installation

1. Clone the repository:

   ```
   git clone <repository-url>
   cd sentiment-analyzer
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app using the safe starter:

   ```
   python start_app.py
   ```

   Or if you prefer a simple start:

   ```
   streamlit run app.py
   ```

2. Enter text in the input field and click "Analyze Sentiment" to get the prediction.

3. Use the advanced options to see detailed explanations and confidence scores.

## Model Training

The sentiment analyzer uses a combined approach:

1. **Machine Learning Models**:

   - Logistic Regression
   - Random Forest
   - Support Vector Machine
   - Ensemble Model (Voting Classifier)

2. **Feature Engineering**:

   - TF-IDF text vectorization
   - VADER sentiment scores
   - Text statistics (exclamations, question marks, etc.)
   - Emoji sentiment analysis

3. **Rule-Based Overrides**: For common idioms and expressions

To train the model on your own data:

```
python train_model.py
```

## Accuracy

The model achieves high accuracy through:

- Advanced preprocessing to handle negation and special cases
- Ensemble modeling to reduce bias and variance
- Rule-based overrides for idiomatic expressions
- Emoji sentiment analysis
- Special handling of negated expressions like "not bad" → positive
- Feature dimension mismatch handling with graceful fallbacks
- Robust error handling for NLTK resources

## Key Features

### Robust Sentiment Detection

- Correctly handles negations like "not bad" (positive) vs "not good" (negative)
- Identifies sentiment in texts with emojis even when words are neutral
- Combines multiple signals (words, emojis, punctuation) for accurate classification

### Error Handling

- Graceful fallbacks for NLTK resource issues
- Feature dimension mismatch detection and handling
- Comprehensive diagnostics and explanations

### Advanced Preprocessing

- Special phrase handling for common expressions
- Emoji sentiment analysis
- Negation scope handling and propagation
- Comprehensive text cleaning and normalization

## License

[MIT License](LICENSE)
