import pandas as pd
import streamlit as st
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import re
import joblib
from sklearn.naive_bayes import MultinomialNB
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
#from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


## function to remove unnecessary text
def clean_text(text):
  text = re.sub(r'http[s]?://\S+', '', text)
  text = re.sub(r'@[A-Za-z0-9]+', '', text)
  text = re.sub(r'[.,!?;]', '', text)
  text = re.sub(r'[^\w\s]', '', text)
  text = re.sub(r'\s+', ' ', text)
  text = re.sub(r'\d+', '', text)
  #text = re.sub(r'[^\w\s]', '',text)
  text = ' '.join(text.split())
  text = text.lower()
  return text

## function to remove stop words
def remove_stopwords(text):
  stop_words = set(stopwords.words('english'))
  words = nltk.word_tokenize(text)
  filtered_words = [word for word in words if word.lower() not in stop_words]
  return " ".join(filtered_words)



# Load your trained model and vectorizer
mnb_model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorize.pkl')
label_encoder = joblib.load('label_encoder.pkl')

## prepare text for prediction
def clean_for_pred(text):
    
    text = clean_text(text)
    text = remove_stopwords(text)
    text = [text]
    x    = vectorizer.transform(text)
    x    =  x.toarray()
    return x

# App appearance
st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="wide")

# Add a title and description
st.title("Sentiment Analysis with Multinomial Naive Bayes Classifier Model")
st.write("""
This app uses a trained Multinomial Naive Bayes (MNB) model to predict the sentiment of a given text. 
The text data used for training the model was obtained from airline tweets. Check kaggler for the raw data.
You can enter a sentence, and the model will classify it as positive or negative sentiment. 
**Contact Jacob Akubire @ jaakubire@gmail.com for anything concerning this app**
""")

# Add some styling
st.markdown("""
<style>
body {
    background-color: #f0f2f6;
    color: #333;
    font-family: 'Arial', sans-serif;
}
header, footer {
    visibility: hidden;
}
.stTextInput > div > div > input {
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 4px;
}
.stButton > button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}
.stButton > button:hover {
    background-color: #45a049;
}
</style>
""", unsafe_allow_html=True)

# Input text
text_input = st.text_input("Enter a sentence for sentiment analysis:")

if text_input:
    # Vectorize the input text
    #text_vectorized = vectorizer.transform([text_input])

    text_clean_vectorized = clean_for_pred(text_input)

    # Predict sentiment
    prediction = mnb_model.predict(text_clean_vectorized)

    # Map prediction to sentiment
    #sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    #sentiment = sentiment_map[prediction[0]]

    sentiment = label_encoder.inverse_transform(prediction)[0]

    # Display the result
    st.write(f"The sentiment of the entered text: {text_input} is **{sentiment}**.")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 50px;">
    <p>Developed by Jacob Akubire</p>
</div>
""", unsafe_allow_html=True)
