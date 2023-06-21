import streamlit as st
import numpy as np
import tensorflow as tf
import transformers
import re
import string
import preprocessor as pre

from transformers import AutoTokenizer
from transformers import TFBertForSequenceClassification

with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True) 

# Preparation model and tokenizer
model_path = "digdoaji/indobert-sentiment-analysis-mandalika-circuit"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = TFBertForSequenceClassification.from_pretrained(model_path)

# Define the maximum sequence length
seq_max_length = 54

# Function to tokenizing input text
def tokenizing_text(sentence):
    sentence = preprocess_text(sentence)
    encoded = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=seq_max_length,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )

    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    return input_ids, attention_mask

# Function to preprocessing input text
def preprocess_text(sentence):
    re_cleansing = "@\S+|https?:\S+|http?:\S|#[A-Za-z0-9]+|^RT[\s]+|(^|\W)\d+"
    for punctuation in string.punctuation:
        sentence = sentence.encode().decode('unicode_escape')
        sentence = re.sub(r'\n', ' ', sentence)
        sentence = pre.clean(sentence)
        sentence = re.sub(r'[^\w\s]', ' ', sentence)
        sentence = re.sub(r'[0-9]', ' ', sentence)
        sentence = re.sub(re_cleansing, ' ', sentence).strip()
        sentence = sentence.replace(punctuation, '')
        sentence = sentence.lower()
    return sentence

# Function to predict sentiment
def predict_sentiment(input_text):
    input_ids, attention_mask = tokenizing_text(input_text)
    prediction = model.predict([input_ids, attention_mask])
    predict_class = np.argmax(prediction.logits).item()
    label_sentiment = {0: "negative", 1: "positive"}
    predict_label = label_sentiment[predict_class]
    return predict_label


# Streamlit web app
def main():
    st.title("Sentiment Analysis of Mandalika International Circuit", anchor=False)
    tweet_text = st.text_area(" ", placeholder="Enter the sentence you want to analyze", label_visibility="collapsed")
    
    if st.button("SEND"):
        if tweet_text.strip() == "":
            st.title("Text Input Still Empty", anchor=False)
            st.info("Please fill in the sentence you want to analyze")
        else:
            sentiment = predict_sentiment(tweet_text)
            if sentiment == "positive":
                st.title("Sentiment Analysis Results", anchor=False)
                st.markdown('<div style="background-color: #5d9c59; padding: 16px; border-radius: 5px; font-weight: bold; color:white;">This sentence contains a positive sentiment</div>', unsafe_allow_html=True)
            elif sentiment == "negative":
                st.title("Sentiment Analysis Results", anchor=False)
                st.markdown('<div style="background-color: #df2e38; padding: 16px; border-radius: 5px; font-weight: bold; color:white;">This sentence contains a negative sentiment</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()