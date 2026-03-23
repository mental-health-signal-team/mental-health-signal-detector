import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")
st.title("Mental Health Signal Detector")
st.write("Enter text to analyze for mental health signals.")
text_input = st.text_area("Input Text", height=200)
model_type = st.selectbox("Select Model", ["lr", "distilbert"])
if st.button("Predict"):
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing... (first request may take up to 30s to wake the server)"):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": text_input, "model_type": model_type},
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()
                label = result["label"]
                probability = result["probability"]
                label_text = "Distress signal detected" if label == 1 else "No distress signal detected"
                color = st.error if label == 1 else st.success
                color(label_text)
                st.metric("Confidence", f"{probability:.0%}")
                st.progress(probability)
            except requests.exceptions.RequestException as e:
                st.error(f"Error: {e}")
