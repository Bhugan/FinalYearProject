import streamlit as st
import PyPDF2
import spacy
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import matplotlib.pyplot as plt
import time
import io

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("SpaCy model 'en_core_web_sm' not found. Please ensure it's installed.")
    st.stop()

# Streamlit UI
st.title("NLP Pipeline Streamlit App")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

# Customizable Parameters
max_tfidf_features = st.sidebar.slider("Max TF-IDF Features", 10, 100, 50)
summary_max_length = st.sidebar.slider("Summarization Max Length", 50, 200, 100)
summary_min_length = st.sidebar.slider("Summarization Min Length", 10, 50, 20)
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

if uploaded_file:
    file_type = uploaded_file.type
    if file_type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif file_type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
    else:
        st.error("Unsupported file type. Please upload a PDF or TXT file.")
        text = None

    if text:
        with st.spinner("Processing..."):
            processed_text = preprocess_text(text)

            # TF-IDF Feature Extraction
            vectorizer = TfidfVectorizer(max_features=max_tfidf_features)
            tfidf_matrix = vectorizer.fit_transform([processed_text])
            tfidf_words = vectorizer.get_feature_names_out()

            # Sentiment Analysis
            sentiment = SentimentIntensityAnalyzer().polarity_scores(processed_text)

            # Summarization
            try:
                summary = pipeline("summarization")(text, max_length=summary_max_length, min_length=summary_min_length, do_sample=False)[0]["summary_text"]
            except Exception as e:
                st.error(f"Error during summarization: {e}")
                summary = "Summarization failed."

            # Visualization - Word Cloud
            wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=wordcloud_max_words).generate(processed_text)

            # Named Entity Recognition Visualization
            ents = [(ent.text, ent.label_) for ent in nlp(text).ents]

        st.success("Processing complete!")

        # Display Results
        st.subheader("Extracted Text")
        st.text_area("", text, height=200)

        st.subheader("Processed Text")
        st.text_area("", processed_text, height=200)

        st.subheader("TF-IDF Features")
        st.write(tfidf_words)

        st.subheader("Sentiment Analysis")
        st.write(f"Positive: {sentiment['pos']}, Negative: {sentiment['neg']}, Neutral: {sentiment['neu']}, Compound: {sentiment['compound']}")

        st.subheader("Summarization")
        st.write(summary)

        st.subheader("Word Cloud")
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)

        st.subheader("Named Entity Recognition")
        st.write(ents)
        #Download buttons
        def download_text(text, filename):
            buffer = io.StringIO(text)
            st.download_button(
                label=f"Download {filename}",
                data=buffer,
                file_name=filename,
                mime="text/plain",
            )
        download_text(text,"Extracted_text.txt")
        download_text(processed_text,"Processed_text.txt")
        download_text(summary,"Summary.txt")

else:
    st.info("Please upload a PDF or TXT file.")
