"""Reusable SpamShield machine-learning backend.

The implementation intentionally preserves the original project's methodology:
regex cleaning, TF-IDF (1, 2)-grams, Logistic Regression, and the strong-keyword
override used before the model prediction.
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

SPAM_THRESHOLD = 0.35
STRONG_SPAM_KEYWORDS = [
    "lottery", "prize", "winner", "claim", "free", "cash", "scan and win",
    "win a chance", "rs", "$", "reward", "credited", "won",
]


def find_dataset_path() -> Path:
    """Find the bundled dataset without depending on the current directory."""
    candidates = (
        Path(__file__).with_name("spam.csv"),
        Path(__file__).parent / "attached_assets" / "spam_1787249827041.csv",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "spam.csv was not found. Keep spam.csv in the project root or upload the dataset."
    )


def load_dataset(path: Optional[str] = None) -> pd.DataFrame:
    """Load and validate the SMS Spam Collection CSV."""
    csv_path = Path(path) if path else find_dataset_path()
    try:
        raw = pd.read_csv(csv_path, encoding="latin-1")
    except Exception as exc:
        raise ValueError(f"Could not read the dataset: {exc}") from exc
    required = {"v1", "v2"}
    if not required.issubset(raw.columns):
        raise ValueError("The dataset must contain the original 'v1' and 'v2' columns.")
    data = raw[["v1", "v2"]].copy()
    data.columns = ["label", "message"]
    data = data.dropna()
    data["label"] = data["label"].map({"ham": 0, "spam": 1})
    data = data.dropna(subset=["label"])
    if data.empty:
        raise ValueError("The dataset contains no usable ham/spam rows.")
    data["label"] = data["label"].astype(int)
    data["message"] = data["message"].astype(str)
    data["clean_message"] = data["message"].apply(clean_text)
    return data


def clean_text(text: Any) -> str:
    """Match the original regex-based preprocessing."""
    text = str(text).lower()
    return re.sub(r"[^a-zA-Z\s]", "", text)


def strong_keyword_detect(text: Any) -> Tuple[bool, Optional[str]]:
    """Return whether an original strong keyword appears and which one matched."""
    cleaned = clean_text(text)
    for keyword in STRONG_SPAM_KEYWORDS:
        if keyword in cleaned:
            return True, keyword
    return False, None


def train_model(path: Optional[str] = None) -> Dict[str, Any]:
    """Train and evaluate the original hybrid model once per Streamlit cache."""
    data = load_dataset(path)
    x_train, x_test, y_train, y_test = train_test_split(
        data["clean_message"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    x_train_vec = vectorizer.fit_transform(x_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train_vec, y_train)

    ml_probabilities = model.predict_proba(vectorizer.transform(x_test))[:, 1]
    predictions = []
    for message, probability in zip(x_test, ml_probabilities):
        is_keyword, _ = strong_keyword_detect(message)
        predictions.append(1 if is_keyword or probability > SPAM_THRESHOLD else 0)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": matrix,
    }
    return {
        "data": data,
        "vectorizer": vectorizer,
        "model": model,
        "metrics": metrics,
        "test_labels": y_test,
        "test_predictions": predictions,
    }


def predict_message(message: Any, bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Predict one message and return UI-friendly explanation fields."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Please enter an SMS message.")
    cleaned = clean_text(message)
    if not cleaned.strip():
        raise ValueError("Please enter a message containing readable text.")
    triggered, keyword = strong_keyword_detect(message)
    probability = float(
        bundle["model"].predict_proba(bundle["vectorizer"].transform([cleaned]))[0][1]
    )
    is_spam = triggered or probability > SPAM_THRESHOLD
    return {
        "is_spam": is_spam,
        "confidence": 1.0 if triggered else (probability if is_spam else 1 - probability),
        "spam_probability": probability,
        "keyword_triggered": triggered,
        "keyword": keyword,
        "reason": (
            f"Strong spam keyword detected: “{keyword}”."
            if triggered
            else "Prediction generated using TF-IDF feature extraction and Logistic Regression."
        ),
    }


def dataset_statistics(data: pd.DataFrame) -> Dict[str, float]:
    total = len(data)
    spam = int(data["label"].sum())
    ham = total - spam
    return {
        "total": total, "spam": spam, "ham": ham,
        "spam_pct": spam / total * 100, "ham_pct": ham / total * 100,
    }