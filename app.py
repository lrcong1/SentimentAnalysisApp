import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained(".")
model = DistilBertForSequenceClassification.from_pretrained(
    "."
)


def predict_sentiment(review):
    inputs = tokenizer(
        review, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=1).item()
    return "Positive" if predicted_class == 1 else "Negative"


# Streamlit app
st.title("Sentiment Analysis")

review = st.text_area("Enter your review:")
if st.button("Analyze"):
    sentiment = predict_sentiment(review)
    st.write(f"Sentiment: {sentiment}")
