import streamlit as st
import pickle

#load pickel

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page config
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

#laveder color box
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #e6e6fa;
}

/* Title */
.title {
    text-align: center;
    font-size: 36px;
    color: #4b0082;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #6a5acd;
    margin-bottom: 30px;
}

/* Input box */
textarea {
    background-color: #f8f8ff !important;
    color: black !important;
    border-radius: 10px !important;
    border: none !important;
    font-size: 16px;
}

/* Button */
.stButton>button {
    background-color: #9370db;
    color: white;
    border-radius: 10px;
    font-weight: bold;
    width: 150px;
}

/* Result */
.result {
    text-align: center;
    font-size: 22px;
    margin-top: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>💬 Sentiment Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analyze the emotion behind your text</div>", unsafe_allow_html=True)

# Input
text = st.text_area("✍️ Enter your text:")

# Button
if st.button("Analyze"):
    if text.strip() != "":
        vector = vectorizer.transform([text])
        prediction = model.predict(vector)[0]

        # Confidence
        confidence = max(model.predict_proba(vector)[0]) * 100

        # Result
        if prediction == 1:
            result = "😊 Positive"
        elif prediction == 0:
            result = "😐 Neutral"
        else:
            result = "😡 Negative"

        st.markdown(f"<div class='result'>{result}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result'>Confidence: {confidence:.2f}%</div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter some text!")
