import re
import string
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = Path(__file__).resolve().parent / "data" / "tickets.csv"
if not DATA_PATH.exists():
    DATA_PATH = Path(__file__).resolve().parent / "tickets.csv"
CONFIDENCE_THRESHOLD = 0.60  # bonus 2: below this -> route to human review
URGENT_KEYWORDS = {
    "down", "urgent", "not working", "immediately", "asap",
    "crash", "crashes", "failed", "broken", "error", "blocked",
}


# ---------------------------------------------------------------------------
# 1. Text preprocessing
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/numbers/extra whitespace, remove stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)          # remove punctuation & numbers
    text = re.sub(r"\s+", " ", text).strip()        # collapse whitespace
    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)


# ---------------------------------------------------------------------------
# 3. Priority tagging (bonus) — simple keyword rule layered on top of the model
# ---------------------------------------------------------------------------
def tag_priority(raw_text: str) -> str:
    lowered = raw_text.lower()
    return "URGENT" if any(kw in lowered for kw in URGENT_KEYWORDS) else "Normal"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
    df["clean_text"] = df["text"].apply(clean_text)
    return df


def train():
    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["category"],
        test_size=0.2, random_state=42, stratify=df["category"]
    )

    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)


    model = MultinomialNB(alpha=0.3)
    model.fit(X_train_vec, y_train)

    # 4. Evaluation literacy
    y_pred = model.predict(X_test_vec)
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\nClassification report (precision / recall / f1 per class):")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix (rows = actual, cols = predicted):")
    labels = sorted(df["category"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels))
    print("=" * 60)

    joblib.dump(model, "model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")
    return model, vectorizer


# ---------------------------------------------------------------------------
# 5. Real-time usability + confidence score + human-review fallback (bonus)
# ---------------------------------------------------------------------------
def classify_ticket(text: str, model, vectorizer) -> dict:
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    probs = model.predict_proba(vec)[0]
    classes = model.classes_

    best_idx = probs.argmax()
    predicted_label = classes[best_idx]
    confidence = probs[best_idx]

    needs_review = confidence < CONFIDENCE_THRESHOLD
    result = {
        "text": text,
        "predicted_category": "Needs Human Review" if needs_review else predicted_label,
        "raw_top_prediction": predicted_label,
        "confidence": round(float(confidence), 3),
        "priority": tag_priority(text),
    }
    return result


def demo(model, vectorizer):

    sample_tickets = [
        "My refund for order #4521 still hasn't arrived after two weeks, please help urgently.",
        "The app keeps crashing every time I open the reports tab, this is broken.",
        "Could you confirm how many paid leave days I have left this quarter?",
        "I'd like to schedule a demo call to see the platform's features.",
        "Hey, just wanted to say the weather has been great lately, thanks!",  # off-topic edge case
    ]

    print("\nLive classification of new sample tickets:\n")
    for t in sample_tickets:
        result = classify_ticket(t, model, vectorizer)
        print(f"Ticket: {t}")
        print(f"  -> Category: {result['predicted_category']} "
              f"(confidence: {result['confidence']:.0%}) | Priority: {result['priority']}")
        print()


REFLECTION_NOTE = """
Reflection:
With more data, I'd add many more real, varied tickets per category (the
current dataset is small and hand-written, so the model leans heavily on a
few obvious keywords). With more time, I'd try a stronger vectorizer/model
combo (e.g. bigrams or Logistic Regression) and cross-validation instead of
a single train/test split, and I'd build a small feedback loop where
human-reviewed tickets get added back into the training set over time.
"""


if __name__ == "__main__":
    trained_model, trained_vectorizer = train()
    demo(trained_model, trained_vectorizer)
    print(REFLECTION_NOTE)
