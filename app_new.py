#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 15:34:03 2025

@author: elifincier
"""

# app.py
import streamlit as st
import pickle
import numpy as np

# Load model
with open("nb_model_alpha.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load vectorizer
with open("tfidf_vect.pkl", "rb") as vect_file:
    tfidf_vect = pickle.load(vect_file)

# Load label encoder
with open("label_encoder.pkl", "rb") as enc_file:
    encoder = pickle.load(enc_file)

# Streamlit UI
st.title("Customer Sentiment Classifier 🚀")

st.write("""
Enter a customer message below — the model will predict the sentiment as:
- Negative
- Neutral
- Positive
""")

user_input = st.text_area("Customer Message:")

if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        # Vectorize input
        input_vect = tfidf_vect.transform([user_input])
        
        # Predict
        pred = model.predict(input_vect)
        
        # Decode label
        sentiment = encoder.inverse_transform(pred)[0]
        
        # Show result
        st.write("### Predicted Sentiment:", sentiment)
    else:
        st.warning("Please enter a message to classify.")
