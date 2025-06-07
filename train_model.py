#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 15:33:15 2025

@author: elifincier
"""

# -*- coding: utf-8 -*-
"""
Model training script — will save model + vectorizer for Streamlit.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import model_selection, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

import pickle

# Load data
output_edit = pd.read_excel('/Users/elifincier/Downloads/output_edited.xlsx')

# Drop unneeded columns
output_edit_2 = output_edit.drop(columns=['Conversation_ID'])

# Train/test split
train_x, test_x, train_y, test_y = model_selection.train_test_split(
    output_edit_2['Customer_Message'], output_edit_2['Sentiment'], random_state=42)

# Label encoding
encoder = preprocessing.LabelEncoder()
encoder.fit(train_y)

train_y_enc = encoder.transform(train_y)
test_y_enc = encoder.transform(test_y)  # IMPORTANT: use SAME encoder

# TF-IDF vectorizer
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3), max_features=10000)
tfidf_vect.fit(output_edit_2['Customer_Message'])

xtrain_tfidf = tfidf_vect.transform(train_x)
xtest_tfidf = tfidf_vect.transform(test_x)

# Grid search for best alpha
params = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]}
grid = GridSearchCV(MultinomialNB(), param_grid=params, scoring='accuracy', cv=5)
grid.fit(xtrain_tfidf, train_y_enc)

print("Best alpha:", grid.best_params_)
print("Best accuracy:", grid.best_score_)

# Final model training
nb_model_alpha = MultinomialNB(alpha=grid.best_params_['alpha'])
nb_model_alpha.fit(xtrain_tfidf, train_y_enc)

# Predictions
predictions_nb_alpha = nb_model_alpha.predict(xtest_tfidf)

# Metrics
print(classification_report(test_y_enc, predictions_nb_alpha, target_names=encoder.classes_))

# Save model
with open("nb_model_alpha.pkl", "wb") as model_file:
    pickle.dump(nb_model_alpha, model_file)

# Save vectorizer
with open("tfidf_vect.pkl", "wb") as vect_file:
    pickle.dump(tfidf_vect, vect_file)

# Save label encoder (so we can decode labels later!)
with open("label_encoder.pkl", "wb") as enc_file:
    pickle.dump(encoder, enc_file)

print("Model, vectorizer, and encoder saved! Ready for Streamlit 🚀")
