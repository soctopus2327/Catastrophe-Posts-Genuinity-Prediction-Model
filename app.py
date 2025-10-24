import streamlit as st
import joblib
import string
import re
import emoji
import nltk
from nltk.stem import PorterStemmer
from scipy.sparse import hstack
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

model = joblib.load("models/lr_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
scaler = joblib.load("models/scaler.pkl") 

st.set_page_config(page_title="Catastrophe Posts Genuinity Prediction", layout="centered")
st.title("🚨 Catastrophe Posts Genuinity Prediction")
st.write(
    "Enter a text from a post to check if it is reporting a **real disaster** or not. "
    "The prediction uses a trained Logistic Regression model with TF-IDF + numeric features."
)

user_input = st.text_area("Enter Tweet Here:")

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"[^a-z\s]", "", text)
    
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a tweet to predict!")
    else:
        cleaned = clean_text(user_input)

        X_text = tfidf.transform([cleaned])

        char_count = len(user_input)
        word_count = len(user_input.split())
        punct_count = len([c for c in user_input if c in string.punctuation])
        X_num = scaler.transform([[char_count, word_count, punct_count]])

        X_combined = hstack([X_text, X_num])

        pred = model.predict(X_combined)[0]
        prob = model.predict_proba(X_combined)[0][pred]

        if pred == 1:
            st.markdown(
                f"<h3 style='color:red;'>✅ Real Disaster Tweet</h3>"
                f"<p>Confidence: {prob:.2f}</p>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<h3 style='color:green;'>ℹ️ Not a Disaster Tweet</h3>"
                f"<p>Confidence: {prob:.2f}</p>",
                unsafe_allow_html=True
            )
