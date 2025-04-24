import os
import pickle
from typing import List, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_PATH = "email_classifier.pkl"

def train_model(data_path: str) -> None:
    """
    Train the email classification model on the provided dataset CSV.
    The CSV is expected to have columns: 'email_body' and 'category'.
    Saves the trained model pipeline to disk.
    """
    df = pd.read_csv(data_path)
    X = df['email_body']
    y = df['category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)

def load_model() -> Pipeline:
    """
    Load the trained model pipeline from disk.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found. Train the model first.")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

def classify_email(model: Pipeline, email_text: str) -> str:
    """
    Classify the email text into a category using the loaded model.
    """
    prediction = model.predict([email_text])
    return prediction[0]
