import streamlit as st
import PyPDF2
import spacy
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import matplotlib.pyplot as plt
import io
import pyperclip  # Import pyperclip for clipboard access

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("SpaCy model 'en_core_web_sm' not found. Please ensure it's installed.")
    st.stop()

# Streamlit UI
st.title("NLP Pipeline Streamlit App")

# File Upload/Clipboard Input
input_source = st.selectbox("Select Input Source", ["Upload File", "Clipboard"])

if input_source == "Upload File":
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
elif input_source == "Clipboard":
    if st.button("Get Text from Clipboard"):
        try:
            text = pyperclip.paste()
            if not text:
                st.warning("Clipboard is empty.")
            else:
                uploaded_file = None #set to none so the rest of the code works as expected.
        except pyperclip.PyperclipException:
            st.error("Clipboard access failed. Please ensure you have pyperclip installed and configured correctly.")
            text = None
    else:
        text = None
        uploaded_file = None #set to none so the rest of the code works as expected.

# Sidebar Options
st.sidebar.header("NLP Options")
analysis_type = st.sidebar.selectbox("Select Analysis", ["Full Analysis", "Tokenized Words", "Sentiment Analysis", "Summarization", "Word Cloud", "Named Entity Recognition"])

# Customizable Parameters (Separate Sliders)
if analysis_type == "Full Analysis" or analysis_type == "Tokenized Words":
    max_tfidf_features = st.sidebar.slider("Max TF-IDF Features", 10, 100, 50)

if analysis_type == "Full Analysis" or analysis_type == "Summarization":
    summary_max_length = st.sidebar.slider("Summarization Max Length", 50, 200, 100)
    summary_min_length = st.sidebar.slider("Summarization Min Length", 10, 50, 20)

if analysis_type == "Full Analysis" or analysis_type == "Word Cloud":
    wordcloud_max_words = st.sidebar.slider("Word Cloud Max Words", 50, 200, 100)

@st.cache_data
def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        return "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

@st.cache_data
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def download_text(text, filename):
    st.download_button(
        label=f"Download {filename}",
        data=text.encode("utf-8"),
        file_name=filename,
        mime="text/plain",
    )

if uploaded_file or (input_source == "Clipboard" and text): #combine input sources.
    if uploaded_file: #handle file upload
        file_type = uploaded_file.type
        if file_type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif file_type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
        else:
            st.error("Unsupported file type. Please upload a PDF or TXT file.")
            text = None

    if text: #process text if we have it from either source.
        processed_text = preprocess_text(text)

        if analysis_type == "Full Analysis":
            # ... (rest of the Full Analysis code) ...
        elif analysis_type == "Tokenized Words":
            vectorizer = TfidfVectorizer(max_features=max_tfidf_features)
            tfidf_matrix = vectorizer.fit_transform([processed_text])
            tfidf_words = vectorizer.get_feature_names_out()
            st.subheader("TF-IDF Features")
            st.write(tfidf_words)
            download_text(" ".join(tfidf_words), "TFIDF_words.txt")

        elif analysis_type == "Sentiment Analysis":
            # ... (rest of the Sentiment Analysis code) ...
        elif analysis_type == "Summarization":
            # ... (rest of the Summarization code) ...
        elif analysis_type == "Word Cloud":
            # ... (rest of the Word Cloud code) ...
        elif analysis_type == "Named Entity Recognition":
            # ... (rest of the Named Entity Recognition code) ...

else:
    st.info("Please upload a file or get text from the clipboard.")
