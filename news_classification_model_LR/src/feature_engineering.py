import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import PROCESSED_DATA_PATH, text_column, label_column

def extract_features(vectorizer=None):
    df = pd.read_csv(PROCESSED_DATA_PATH)

    if vectorizer is None:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        X = vectorizer.fit_transform(df[text_column])
    else:
        X = vectorizer.transform(df[text_column])

    y = df[label_column]
    return X, y, vectorizer


if __name__=="__main__":
    extract_features()

# imported feature extraction - tfidf vectorizer: to 
# imported the data paths for processed dataset, text column of dataset and category column of dataset

# created a method extract_features to extract:
# 1. X- numerical feature matrix of the text
# 2. y- category labels -target
# 3. vectorizer- actual feature extraction method- text_to_number
