import pickle
import os

# Define the feature columns that should be present
feature_columns = [
    'vader_compound', 'vader_pos', 'vader_neu', 'vader_neg', 
    'exclamation_count', 'question_count', 'caps_ratio', 'text_length',
    'positive_emoji_count', 'negative_emoji_count', 'emoji_sentiment_ratio',
    'negation_count'
]

# Check if the file exists
if not os.path.exists('model/feature_columns.pkl'):
    print("Creating feature_columns.pkl file...")
    with open('model/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    print("Done!")
else:
    print("feature_columns.pkl already exists.")
