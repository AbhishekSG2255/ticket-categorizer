import streamlit as st
from ticket_classifier import train, classify_ticket, CONFIDENCE_THRESHOLD

st.set_page_config(page_title="Ticket Categorizer", page_icon="🎫")
st.title("🎫 Auto Ticket Categorizer")
st.caption("TF-IDF + Naive Bayes · classifies a ticket as Billing / Technical / HR / General")


@st.cache_resource
def get_model():
    return train()


model, vectorizer = get_model()

ticket_text = st.text_area(
    "Paste a support ticket (subject + body):",
    placeholder="e.g. My refund still hasn't arrived after two weeks, this is urgent.",
    height=120,
)

if st.button("Classify ticket", type="primary") and ticket_text.strip():
    result = classify_ticket(ticket_text, model, vectorizer)

    if result["confidence"] < CONFIDENCE_THRESHOLD:
        st.warning(
            f"⚠️ Needs Human Review — confidence too low "
            f"({result['confidence']:.0%}), closest guess: {result['raw_top_prediction']}"
        )
    else:
        st.success(f"Category: **{result['predicted_category']}**  ·  Confidence: {result['confidence']:.0%}")

    priority_color = "🔴" if result["priority"] == "URGENT" else "🟢"
    st.write(f"Priority: {priority_color} **{result['priority']}**")

    st.progress(result["confidence"])
elif ticket_text.strip() == "":
    st.info("Type or paste a ticket above, then click Classify.")
