import sys
import os
import argparse
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_FILE = "sentiment_model.joblib"
DATA_FILE = "reviews_0-250.csv"
SAMPLE_DATA_FILE = "reviews_sample.csv"

def get_data_file():
    if os.path.exists(DATA_FILE):
        return DATA_FILE
    return SAMPLE_DATA_FILE

def train_model(csv_path=None, model_path=MODEL_FILE, sample_size=100000):
    if csv_path is None:
        csv_path = get_data_file()
    df = pd.read_csv(csv_path, usecols=["rating", "review_text"], nrows=sample_size).dropna()
    def map_rating(val):
        if val >= 4:
            return "Positive"
        if val <= 2:
            return "Negative"
        return "Neutral"
    df["sentiment"] = df["rating"].apply(map_rating)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=25000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=200))
    ])
    pipeline.fit(df["review_text"], df["sentiment"])
    joblib.dump(pipeline, model_path, compress=3)
    return pipeline

def load_model(model_path=MODEL_FILE):
    if not os.path.exists(model_path):
        return train_model(get_data_file(), model_path)
    return joblib.load(model_path)

def predict(model, text):
    pred = model.predict([text])[0]
    prob = max(model.predict_proba([text])[0]) * 100
    return pred, prob

def interactive_mode(model):
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in ("exit", "quit", "q"):
            break
        sentiment, conf = predict(model, text)
        print(f"{sentiment} ({conf:.1f}%)")

def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("text", nargs="*", help=argparse.SUPPRESS)
    parser.add_argument("--train", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.train:
        train_model()
        return

    model = load_model()

    if args.text:
        query = " ".join(args.text).strip()
        sentiment, conf = predict(model, query)
        print(f"{sentiment} ({conf:.1f}%)")
    else:
        interactive_mode(model)

if __name__ == "__main__":
    main()
