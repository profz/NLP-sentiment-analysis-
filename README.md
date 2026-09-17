# NLP Sentiment Analyzer

<html>
  <h2 align="center">
    <img src="pngfind.com-mca-logo-png-6131138.png" width="250"/>
  </h2>
</html>

A minimal and lightweight CLI-based sentiment analysis tool trained on customer reviews. Classifies text as Positive, Neutral, or Negative with confidence scores.

## Requirements

- Python 3.8+
- pandas
- scikit-learn
- joblib

Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Single Review Analysis (CLI)
```bash
python sentiment.py "This product exceeded all my expectations, absolutely love it!"
```
Output:
```text
Positive (99.8%)
```

```bash
python sentiment.py "Broke after two days, completely useless."
```
Output:
```text
Negative (98.6%)
```

### 2. Interactive Mode
```bash
python sentiment.py
```
Output:
```text
> Excellent quality and fast shipping!
Positive (99.7%)
> It is okay, nothing special.
Neutral (89.4%)
> exit
```

### 3. Retraining the Model
Trains automatically using `reviews_sample.csv` (or `reviews_0-250.csv` if present):
```bash
python sentiment.py --train
```

### 4. Python API Snippet
```python
from sentiment import load_model, predict

model = load_model()
sentiment, confidence = predict(model, "Best purchase I have made!")
print(f"Sentiment: {sentiment}, Confidence: {confidence:.1f}%")
```

## Dataset

- `reviews_sample.csv`: Lightweight balanced sample dataset (7,500 rows, ~3.4 MB) included for testing and fast training.
- `reviews_0-250.csv`: Full dataset (~600,000 rows, ignored by `.gitignore` due to large file size).
