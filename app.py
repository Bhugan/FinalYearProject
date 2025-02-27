import streamlit as st
import PyPDF2
import spacy
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import matplotlib.pyplot as plt
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
        processed_text = preprocess_text(text)

        if analysis_type == "Full Analysis":
            with st.spinner("Processing..."):
                # TF-IDF Feature Extraction
                vectorizer = TfidfVectorizer(max_features=max_tfidf_features)
                tfidf_matrix = vectorizer.fit_transform([processed_text])
                tfidf_words = vectorizer.get_feature_names_out()

                # Sentiment Analysis
                sentiment = SentimentIntensityAnalyzer().polarity_scores(processed_text)

                # Summarization
                try:
                    summary_result = pipeline("summarization")(text, max_length=summary_max_length, min_length=summary_min_length, do_sample=False)
                    summary = summary_result[0]["summary_text"]
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
            download_text(text, "Extracted_text.txt")

            st.subheader("Processed Text")
            st.text_area("", processed_text, height=200)
            download_text(processed_text, "Processed_text.txt")

            st.subheader("TF-IDF Features")
            st.write(tfidf_words)
            download_text(" ".join(tfidf_words), "TFIDF_words.txt")

            st.subheader("Sentiment Analysis")
            st.write(f"Positive: {sentiment['pos']}, Negative: {sentiment['neg']}, Neutral: {sentiment['neu']}, Compound: {sentiment['compound']}")
            download_text(str(sentiment), "Sentiment.txt")

            st.subheader("Summarization")
            st.write(summary)
            download_text(summary, "Summary.txt")

            st.subheader("Word Cloud")
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)

            st.subheader("Named Entity Recognition")
            st.write(ents)
            download_text(str(ents), "Named_Entities.txt")

        elif analysis_type == "Tokenized Words":
            vectorizer = TfidfVectorizer(max_features=max_tfidf_features)
            tfidf_matrix = vectorizer.fit_transform([processed_text])
            tfidf_words = vectorizer.get_feature_names_out()
            st.subheader("TF-IDF Features")
            st.write(tfidf_words)
            download_text(" ".join(tfidf_words), "TFIDF_words.txt")

        elif analysis_type == "Sentiment Analysis":
            sentiment = SentimentIntensityAnalyzer().polarity_scores(processed_text)
            st.subheader("Sentiment Analysis")
            st.write(f"Positive: {sentiment['pos']}, Negative: {sentiment['neg']}, Neutral: {sentiment['neu']}, Compound: {sentiment['compound']}")
            download_text(str(sentiment), "Sentiment.txt")

        elif analysis_type == "Summarization":
            try:
                summary_result = pipeline("summarization")(text, max_length=summary_max_length, min_length=summary_min_length, do_sample=False)
                summary = summary_result[0]["summary_text"]
            except Exception as e:
                st.error(f"Error during summarization: {e}")
                summary = "Summarization failed."
            st.subheader("Summarization")
            st.write(summary)
            download_text(summary, "Summary.txt")

        elif analysis_type == "Word Cloud":
            wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=wordcloud_max_words).generate(processed_text)
            st.subheader("Word Cloud")
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)

        elif analysis_type == "Named Entity Recognition":
            ents = [(ent.text, ent.label_) for ent in nlp(text).ents]
            st.subheader("Named Entity Recognition")
            st.write(ents)
            download_text(str(ents), "Named_Entities.txt")

else:
    st.info("Please upload a PDF or TXT file.")
