import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
import re
import string
import preprocessor as pre


from transformers import AutoTokenizer
from transformers import TFBertForSequenceClassification
from tensorflow import keras

with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True) 

# Preparation model
model_path = "D:\LuKanAnakIT\Menuju Tugas Akhir\Tugas Akhir\@Penulisan\Coding\Prototipe1\IndoBERT_Model_M54.h5"
bertModel_package = 'indobenchmark/indobert-base-p2'
tokenizer = AutoTokenizer.from_pretrained(bertModel_package, do_lower_case=True)
keras.utils.get_custom_objects().update({'TFBertForSequenceClassification': TFBertForSequenceClassification})

# load model on first launch
st.cache_data()
model = keras.models.load_model(model_path)

# Function to tokenizing input text
seq_max_length = 54

def tokenizing_data(data):
    for sentence in data:
        sentence = preprocess_text(sentence)
        encoded = tokenizer.encode_plus(
            data,
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
    for punctuation in string.punctuation:
        re_cleansing = "@\S+|https?:\S+|http?:\S|#[A-Za-z0-9]+|^RT[\s]+|(^|\W)\d+"
        sentence = sentence.encode().decode('unicode_escape')
        sentence = sentence.lower()
        sentence = re.sub(r'\n', ' ', sentence)
        sentence = pre.clean(sentence)
        sentence = re.sub(r'[^\w\s]', ' ', sentence)
        sentence = re.sub(r'[0-9]', ' ', sentence)
        sentence = re.sub(re_cleansing, ' ', sentence).strip()
        sentence = sentence.replace(punctuation, '')
    return sentence

# Function to predict sentiment
def predict_sentiment(input_text):
    input_ids, attention_mask = tokenizing_data(input_text)
    prediction = model.predict([input_ids, attention_mask])
    predict_label = np.round(prediction).flatten()
    label_sentiment = {0: "negatif", 1: "positif"}
    predict_label = label_sentiment[int(predict_label)]
    return predict_label

# Streamlit web app
def main():
    st.title("Analisis Sentimen Sirkuit Internasional Mandalika", anchor=False)
    tweet_text = st.text_area(" ", placeholder="Masukkan kalimat yang ingin dianalisis", label_visibility="collapsed")
    
    if st.button("KIRIM"):
        if tweet_text.strip() == "":
            st.title("Input Teks Kosong", anchor=False)
            st.info("Mohon isi kalimat yang ingin dianalisis")
        else:
            sentiment = predict_sentiment(tweet_text)
            if sentiment == "positif":
                st.title("Hasil Analisis Sentimen", anchor=False)
                st.markdown('<div style="background-color: #5d9c59; padding: 16px; border-radius: 5px; font-weight: bold; color:white;">Kalimat tersebut mengandung sentimen positif</div>', unsafe_allow_html=True)
            if sentiment == "negatif":
                st.title("Hasil Analisis Sentimen", anchor=False)
                st.markdown('<div style="background-color: #df2e38; padding: 16px; border-radius: 5px; font-weight: bold; color:white;">Kalimat tersebut mengandung sentimen negatif</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()