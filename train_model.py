import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from utils.preprocess import clean_text, extract_features
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    print("📊 Loading dataset...")
    df = pd.read_csv("data/sentiment140.csv", encoding='latin-1', header=None)
    df.columns = ['target', 'id', 'date', 'query', 'user', 'text']
    df = df[['text', 'target']]
    
    # Convert to binary: 0 = negative, 4 = positive
    df['target'] = df['target'].apply(lambda x: 1 if x == 4 else 0)
    
    # Remove duplicates and null values
    df = df.drop_duplicates(subset=['text'])
    df = df.dropna()
    
    print(f"📈 Dataset size: {len(df)} samples")
    print(f"📊 Class distribution:")
    print(df['target'].value_counts())
    
    return df

def create_enhanced_features(df):
    """Create enhanced features for better accuracy"""
    print("🔧 Creating enhanced features...")
    
    # Clean text
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Add VADER sentiment scores as features
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = df['text'].apply(lambda x: sia.polarity_scores(x))
    
    df['vader_compound'] = [score['compound'] for score in sentiment_scores]
    df['vader_pos'] = [score['pos'] for score in sentiment_scores]
    df['vader_neu'] = [score['neu'] for score in sentiment_scores]
    df['vader_neg'] = [score['neg'] for score in sentiment_scores]
    
    # Add text features
    feature_data = df['text'].apply(extract_features)
    feature_df = pd.DataFrame(list(feature_data))
    
    df = pd.concat([df, feature_df], axis=1)
    
    return df

def train_models(X_train, X_test, y_train, y_test, additional_features_train, additional_features_test):
    """Train multiple models and select the best one"""
    print("🤖 Training models...")
    
    models = {
        'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
        
        print(f"{name} - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_name = name
    
    print(f"\n🏆 Best model: {best_name} with accuracy: {best_score:.4f}")
    
    # Create ensemble model
    print("🔄 Creating ensemble model...")
    ensemble = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(probability=True, random_state=42))
        ],
        voting='soft'
    )
    
    ensemble.fit(X_train, y_train)
    ensemble_pred = ensemble.predict(X_test)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    
    print(f"🎯 Ensemble accuracy: {ensemble_accuracy:.4f}")
    
    # Choose final model (ensemble if better, otherwise best individual)
    if ensemble_accuracy > best_score:
        final_model = ensemble
        final_name = "Ensemble"
        final_pred = ensemble_pred
        print("✅ Using ensemble model as final model")
    else:
        final_model = best_model
        final_name = best_name
        final_pred = results[best_name]['predictions']
        print(f"✅ Using {best_name} as final model")
    
    return final_model, final_pred, results

def evaluate_model(y_test, y_pred, model_name):
    """Evaluate and visualize model performance"""
    print(f"\n📊 Evaluation Results for {model_name}:")
    print("="*50)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'model/confusion_matrix_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Create enhanced features
    df = create_enhanced_features(df)
    
    # Prepare features
    # TF-IDF vectorizer with optimized parameters
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),  # Include bigrams
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    X_text = vectorizer.fit_transform(df['clean_text'])
    
    # Additional features
    additional_features = df[['vader_compound', 'vader_pos', 'vader_neu', 'vader_neg', 
                             'exclamation_count', 'question_count', 'caps_ratio', 'text_length']].fillna(0)
    
    y = df['target']
    
    # Split data
    X_train_text, X_test_text, X_train_add, X_test_add, y_train, y_test = train_test_split(
        X_text, additional_features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Combine text and additional features
    from scipy.sparse import hstack
    X_train = hstack([X_train_text, X_train_add.values])
    X_test = hstack([X_test_text, X_test_add.values])
    
    # Train models
    final_model, y_pred, all_results = train_models(X_train, X_test, y_train, y_test, X_train_add, X_test_add)
    
    # Evaluate final model
    evaluate_model(y_test, y_pred, "Final_Model")
    
    # Save model and vectorizer
    print("\n💾 Saving model and components...")
    with open("model/sentiment_model.pkl", "wb") as f:
        pickle.dump(final_model, f)
    
    with open("model/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    # Save feature columns for consistency
    with open("model/feature_columns.pkl", "wb") as f:
        pickle.dump(additional_features.columns.tolist(), f)
    
    print("✅ Model training completed successfully!")
    print(f"📈 Final model accuracy: {accuracy_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    main()
