import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download("stopwords")

st.set_page_config(
    page_title="AI Fake News Detection",
    page_icon="📰",
    layout="centered"
)

model = joblib.load("best_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

st.title("📰 AI Fake News Detection")
st.markdown("---")

st.write(
    "This application uses Machine Learning and Natural Language Processing "
    "to classify news articles as **REAL** or **FAKE**."
)

news = st.text_area(
    "Paste News Article",
    height=220
)

if st.button("🔍 Predict", use_container_width=True):

    if news.strip() == "":
        st.warning("Please enter a news article.")
    else:
        cleaned = clean_text(news)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        st.markdown("---")

        if prediction == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")

st.markdown("---")
st.caption("Developed using Python, Scikit-learn, Streamlit and NLP")
