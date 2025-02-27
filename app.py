import streamlit as st
import PyPDF2
import spacy
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import matplotlib.pyplot as plt

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Streamlit UI
st.title("NLP Pipeline Streamlit App")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    return "".join([page.extract_text() for page in reader.pages if page.extract_text()])

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    st.subheader("Extracted Text")
    st.text_area("", text, height=200)

    processed_text = preprocess_text(text)
    st.subheader("Processed Text")
    st.text_area("", processed_text, height=200)

    # TF-IDF Feature Extraction
    vectorizer = TfidfVectorizer(max_features=50)
    tfidf_matrix = vectorizer.fit_transform([processed_text])
    tfidf_words = vectorizer.get_feature_names_out()
    st.subheader("TF-IDF Features")
    st.write(tfidf_words)

    # Sentiment Analysis
    sentiment = SentimentIntensityAnalyzer().polarity_scores(processed_text)
    st.subheader("Sentiment Analysis")
    st.write(sentiment)

    # Summarization
    summary = pipeline("summarization")(text, max_length=100, min_length=20, do_sample=False)[0]["summary_text"]
    st.subheader("Summarization")
    st.write(summary)

    # Visualization - Word Cloud
    st.subheader("Word Cloud")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(processed_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

    # Named Entity Recognition Visualization
    st.subheader("Named Entity Recognition")
    ents = [(ent.text, ent.label_) for ent in nlp(text).ents]
    st.write(ents)
