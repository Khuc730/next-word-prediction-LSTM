import streamlit as st
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load model
model = load_model("lstm_model_2.h5")


# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


# Load max length
with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)


def predict_next_word(text):

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=max_len-1,
        padding="pre"
    )

    prediction = model.predict(padded)

    predicted_index = np.argmax(prediction)


    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return "Not Found"



# Streamlit UI

st.title("🧠 Next Word Prediction using LSTM")


text = st.text_input("Enter a sentence")


if st.button("Predict"):

    if text:

        result = predict_next_word(text)

        st.success(
            f"Next word: {result}"
        )

    else:
        st.warning("Enter some text")