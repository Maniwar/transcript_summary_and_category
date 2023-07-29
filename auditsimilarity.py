import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import numpy as np
import xlsxwriter
import chardet
from transformers import pipeline
import base64
from io import BytesIO
import streamlit as st
import textwrap
from categories_josh1 import default_categories

# Set page title and layout
st.set_page_config(page_title="👨‍💻 Transcript Categorization")

# Initialize BERT model
@st.cache_resource
def initialize_bert_model():
    return SentenceTransformer('paraphrase-MiniLM-L12-v2')

# Create a dictionary to store precomputed embeddings
@st.cache_resource
def compute_keyword_embeddings(keywords):
    model = initialize_bert_model()
    keyword_embeddings = {}
    for keyword in keywords:
        keyword_embeddings[keyword] = model.encode([keyword])[0]
    return keyword_embeddings

# Function to preprocess the text
@st.cache_data
def preprocess_text(text):
    # Convert to string if input is a float
    if isinstance(text, float):
        text = str(text)

    # Remove unnecessary characters and weird characters
    text = text.encode('ascii', 'ignore').decode('utf-8')

    # Return the text without removing stop words
    return text

# Function to perform sentiment analysis
@st.cache_data
def perform_sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    return compound_score

# Function to summarize the text
@st.cache_resource
def summarize_text(text):
    summarization_pipeline = pipeline("summarization", model="knkarthick/MEETING_SUMMARY")
    summary = summarization_pipeline(text,  max_length=100, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Function to summarize large texts
def summarize_large_text(text):
    MAX_TOKENS = 1024  # Maximum number of tokens that BART can handle
    MIN_TOKENS_FOR_SUMMARY = 100  # Any text longer than this will be summarized
    tokenized_text = nltk.word_tokenize(text)

    if len(tokenized_text) > MIN_TOKENS_FOR_SUMMARY:
        chunks = textwrap.wrap(text, width=MAX_TOKENS, break_long_words=False)
        summarized_chunks = [summarize_text(chunk) for chunk in chunks]
        return ' '.join(summarized_chunks)
    else:
        return text

# Function to compute semantic similarity
def compute_semantic_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Streamlit interface
st.title("👨‍💻 Transcript Categorization")

# Add checkbox for emerging issue mode
emerging_issue_mode = st.sidebar.checkbox("Emerging Issue Mode")

# Sidebar description for emerging issue mode
st.sidebar.write("Emerging issue mode allows you to set a minimum similarity score. If the comment doesn't match up to the categories based on the threshold, it will be set to NO MATCH.")

# Add slider for semantic similarity threshold in emerging issue mode
similarity_threshold = None
similarity_score = None
best_match_score = None

if emerging_issue_mode:
    similarity_threshold = st.sidebar.slider("Semantic Similarity Threshold", min_value=0.0, max_value=1.0, value=0.35)

# Edit categories and keywords
st.sidebar.header("Edit Categories")

keywords_text = {}
keyword_embeddings = {}

for category in default_categories:
    keywords_text[category] = st.sidebar.text_area(f"Keywords for {category} (One per line)", "\n".join(default_categories[category]))
    keywords_text[category] = [keyword.strip() for keyword in keywords_text[category].split("\n")]
    keyword_embeddings.update(compute_keyword_embeddings(keywords_text[category]))

# Convert keywords to lowercase
keyword_embeddings = {keyword.lower(): embedding for keyword, embedding in keyword_embeddings.items()}

# Get user input comment
comment = st.text_area("Enter your comment here")

# Summarize the comment if necessary
summarized_comment = summarize_large_text(comment)

# Display the summarized comment
st.subheader("Summarized Comment")
st.write(summarized_comment)

# Preprocess the comment
comment = preprocess_text(summarized_comment)

# Encode the comment
model = initialize_bert_model()
comment_embedding = model.encode([comment])[0]

# Compute similarity scores and store them in a DataFrame
similarity_scores = []
best_match_score = 0
best_match = None

for main_category, keywords in keywords_text.items():
    for keyword in keywords:
        keyword_embedding = keyword_embeddings[keyword.lower()]
        similarity_score = compute_semantic_similarity(keyword_embedding, comment_embedding)
        similarity_scores.append({'Keyword': keyword, 'Similarity Score': similarity_score})
        if similarity_score > best_match_score:
            best_match_score = similarity_score
            best_match = keyword

# Check if the best match is under the threshold
if emerging_issue_mode and best_match_score < similarity_threshold:
    best_match = "NO MATCH"

# Show the best match and its score
st.subheader("Best Match")
st.write(best_match)

# Show the score of the best match
st.subheader("Best Match Score")
st.write(best_match_score)

# Create a DataFrame with similarity scores
df_similarity_scores = pd.DataFrame(similarity_scores)

# Get the top 10 items based on similarity score
top_10_items = df_similarity_scores.nlargest(10, 'Similarity Score')

# Display the top 10 items
st.subheader("Top 10 Items")
st.dataframe(top_10_items)

# Print similarity scores
st.subheader("Similarity Scores")
st.dataframe(similarity_scores)
