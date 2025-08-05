# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 18:17:04 2025

@author: bonni
"""

import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack

# Load model and vectorizer
model = joblib.load("Multinomial_NB_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

def preprocess_input(sender, subject, body, url_count, vectorizer):
    # Combine text fields exactly like in training
    combined_text = f"{sender} {body} {subject}"
    # Transform text to TF-IDF vector
    text_vector = vectorizer.transform([combined_text])
    # Convert URL count to array
    url_feature = np.array([[url_count]])
    # Combine TF-IDF features with URL numeric feature
    final_vector = hstack((text_vector, url_feature))
    return final_vector

# Streamlit UI
st.title("Phishing Email Detection")
st.markdown("Enter email info below to predict phishing likelihood:")

sender = st.text_input("Sender")
subject = st.text_input("Subject")
body = st.text_area("Body")
url_count = st.number_input("Number of URLs in email", min_value=0, max_value=100, step=1)

if st.button("Predict"):
    # Preprocess inputs
    features = preprocess_input(sender, subject, body, url_count, vectorizer)
    # Predict with the model
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]  # Probability of phishing

    if prediction == 1:
        st.error(f"⚠️ Prediction: Phishing email detected with {probability:.2%} confidence.")
    else:
        st.success(f"✅ Prediction: Not phishing with {1 - probability:.2%} confidence.")
