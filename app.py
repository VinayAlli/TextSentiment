import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from transformers import pipeline
import torch
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding

# --------- SENTIMENT ANALYSIS FUNCTIONS ---------
@st.cache_resource
def load_sentiment_model():
    # Define a custom SimpleRNN if required by the saved model.
    from tensorflow.keras.layers import SimpleRNN as OriginalSimpleRNN
    class CustomSimpleRNN(OriginalSimpleRNN):
        def __init__(self, *args, **kwargs):
            kwargs.pop('time_major', None)
            super().__init__(*args, **kwargs)
    # Load the trained model (simple_rnn_imdb.h5)
    model = load_model('simple_rnn_imdb.h5', custom_objects={'SimpleRNN': CustomSimpleRNN})
    return model

@st.cache_data
def load_word_index():
    # Load the IMDb dataset word index
    word_index = imdb.get_word_index()
    return word_index

def preprocess_text(text, word_index):
    words = text.lower().split()
    # Unknown words are assigned index 2 and we shift by 3 as done in training.
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(text, model, word_index):
    preprocessed = preprocess_text(text, word_index)
    prediction = model.predict(preprocessed)
    sentiment = "Positive" if prediction[0][0] > 0.7 else "Negative"
    return sentiment, prediction[0][0]

# --------- TEXT GENERATION (RAG-inspired) FUNCTIONS ---------
@st.cache_data
def load_text_generator():
    # Load GPT-2 for text generation (simulate a RAG explanation)
    text_generator = pipeline("text-generation", model="gpt2")
    return text_generator

def generate_explanation(review_text, generator):
    prompt = "Review: " + review_text + "\nSentiment:"
    generated = generator(prompt, max_length=150, num_return_sequences=1,temperature=0.7,
        top_p=0.9,         
        top_k=50, 
        no_repeat_ngram_size=2,   
        repetition_penalty=1.2  )
    explanation = generated[0]["generated_text"]
    return explanation

# --------- EMBEDDING DEMO FUNCTIONS ---------
def embedding_demo(sentence, voc_size=10000, sent_length=8, dim=10):
    # One-hot encode the sentence
    one_hot_repr = one_hot(sentence, voc_size)
    padded_seq = pad_sequences([one_hot_repr], maxlen=sent_length, padding='pre')
    # Build a simple embedding model
    model = Sequential()
    model.add(Embedding(voc_size, dim, input_length=sent_length))
    model.compile('adam', 'mse')
    embeddings = model.predict(padded_seq)
    return padded_seq, embeddings

# --------- STREAMLIT APP STRUCTURE ---------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a Demo", 
                            ["Sentiment Analysis", "Text Generation Explanation", "Embedding Demo"])

if page == "Sentiment Analysis":
    st.title("Movie Review Sentiment Analysis")
    st.write("Enter a movie review and get the sentiment prediction (Positive/Negative).")
    review_input = st.text_area("Enter your movie review here:", height=150)
    if st.button("Analyze Sentiment"):
        if review_input.strip() == "":
            st.error("Please enter a valid review!")
        else:
            model = load_sentiment_model()
            word_index = load_word_index()
            sentiment, score = predict_sentiment(review_input, model, word_index)
            st.subheader("Prediction Result")
            st.write("Sentiment:", sentiment)
            st.write("Prediction Score:", score)

elif page == "Text Generation Explanation":
    st.title("Text Generation Explanation (RAG-inspired)")
    st.write("Enter a movie review to generate a text explanation using GPT-2.")
    review_input = st.text_area("Enter your movie review here:", height=150)
    if st.button("Generate Explanation"):
        if review_input.strip() == "":
            st.error("Please enter a valid review!")
        else:
            generator = load_text_generator()
            explanation = generate_explanation(review_input, generator)
            st.subheader("Generated Explanation")
            st.write(explanation)

elif page == "Embedding Demo":
    st.title("Word Embedding Demo")
    st.write("Enter a sentence to view its one-hot padded representation and embedding vectors.")
    sentence_input = st.text_input("Enter a sentence", value="the glass of milk")
    if st.button("Show Embedding"):
        padded_seq, embeddings = embedding_demo(sentence_input)
        st.subheader("Padded Sequence (One-hot Indices)")
        st.write(padded_seq)
        st.subheader("Embedding Vectors")
        st.write(embeddings[0])
