# Auto Email / Ticket Categorizer

NLP classifier that reads a support ticket (subject + body) and predicts its
department: **Billing, Technical, HR, or General**.

## How it works
1. **Preprocessing** — lowercase, strip punctuation/numbers, remove stopwords.
2. **Feature representation** — TF-IDF (unigrams + bigrams).
3. **Model** — Multinomial Naive Bayes (fast, strong baseline for short
   word-frequency-driven text, and gives usable probability scores).
4. **Evaluation** — accuracy, precision/recall/F1 per class, confusion matrix.
5. **Live classification** — `classify_ticket()` scores any new ticket on demand.

## Bonus objectives included
- ✅ Confidence score returned with every prediction
- ✅ Tickets below 60% confidence are routed to "Needs Human Review" instead of auto-assigned
- ✅ Keyword-based URGENT / Normal priority tagging
- ✅ Mini live demo — Streamlit app where you type a ticket and get an instant result
- ✅ Reflection note (printed at the end of `ticket_classifier.py`)

## Run it

```bash
pip install -r requirements.txt

# Train, evaluate, and see 5 sample predictions in the terminal:
python ticket_classifier.py

# Interactive live demo:
streamlit run app.py
```

## Approach summary (for the submission form)
Used TF-IDF (unigrams+bigrams) with Multinomial Naive Bayes for ticket
classification. Predictions include a confidence score; anything under 60%
confidence is routed to a "Needs Human Review" queue instead of being
auto-assigned, since a wrong silent routing is worse than asking a human.
Added a simple keyword-based urgent/normal priority tag on top of the
category prediction, and a Streamlit UI for live single-ticket testing.

## Files
- `data/tickets.csv` — dummy labeled dataset (48+ tickets across 4 categories)
- `ticket_classifier.py` — preprocessing, training, evaluation, live classification
- `app.py` — Streamlit live demo
- `requirements.txt`
