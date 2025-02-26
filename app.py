pip install -r requirements.txt
python -m spacy download en_core_web_sm
import streamlit as st
import PyPDF2
import nltk
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from transformers import pipeline
import spacy
import subprocess

# Check if the SpaCy model is installed, otherwise install it
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Streamlit UI
st.title("NLP Pipeline Streamlit App")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

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
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(processed_text)
    st.subheader("Sentiment Analysis")
    st.write(sentiment)
    
    # Summarization
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=100, min_length=20, do_sample=False)[0]["summary_text"]
    st.subheader("Summarization")
    st.write(summary)
    
    # Visualization - Word Cloud
    st.subheader("Word Cloud")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(processed_text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)
    
    # Named Entity Recognition Visualization
    st.subheader("Named Entity Recognition")
    doc = nlp(text)
    ents = [(ent.text, ent.label_) for ent in doc.ents]
    st.write(ents)
