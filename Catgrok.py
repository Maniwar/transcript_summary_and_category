import os
import time
import math
import base64
from io import BytesIO
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import chardet
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st
import textwrap
from categories_josh_sub_V6_3 import default_categories
import json
from tqdm import tqdm
import re
import string
import unicodedata
from datetime import datetime
from collections import defaultdict

# Set environment variable for tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)

# --- Device Detection ---
def get_device():
    """Determine the best available device for computation."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

# --- Model Initialization ---
@st.cache_resource
def initialize_bert_model():
    """Initialize and cache the BERT model for embedding generation."""
    start_time = time.time()
    device = get_device()
    model = SentenceTransformer('all-mpnet-base-v2', device=device)
    print(f"BERT model initialized on {device}. Time taken: {time.time() - start_time:.2f} seconds.")
    return model

@st.cache_resource
def get_summarization_model_and_tokenizer():
    """Initialize and cache the summarization model and tokenizer."""
    model_name = "knkarthick/MEETING_SUMMARY"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = get_device()
    model.to(device)
    return model, tokenizer, device

@st.cache_resource
def get_sentiment_model():
    """Initialize and cache the sentiment analysis model."""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)

# --- Utility Functions ---
def preprocess_text(text):
    """Preprocess text by removing special characters and normalizing whitespace."""
    if pd.isna(text):
        return ""
    text = str(text).encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.replace('\n', ' ').replace('\r', '').replace(' ', ' ')
    return re.sub(r'\s+', ' ', text).strip()

def perform_sentiment_analysis(text, sentiment_model=None, use_nltk=True):
    """Perform sentiment analysis using NLTK or a transformer model."""
    if not isinstance(text, str):
        return 0.0
    if use_nltk:
        analyzer = SentimentIntensityAnalyzer()
        return analyzer.polarity_scores(text)['compound']
    else:
        try:
            result = sentiment_model(text[:512])[0]
            return result['score'] if result['label'] == 'POSITIVE' else -result['score']
        except Exception:
            return 0.0

def get_token_count(text, tokenizer):
    """Count tokens in the text using the provided tokenizer."""
    try:
        return len(tokenizer.encode(text)) - 2
    except Exception:
        return 0

def split_comments_into_chunks(comments, tokenizer, max_tokens=1000):
    """Split comments into chunks based on token limits."""
    sorted_comments = sorted(comments, key=lambda x: x[1], reverse=True)
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0
    for comment, tokens in sorted_comments:
        if tokens > max_tokens:
            parts = textwrap.wrap(comment, width=max_tokens)
            for part in parts:
                part_tokens = get_token_count(part, tokenizer)
                if current_chunk_tokens + part_tokens > max_tokens:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [part]
                    current_chunk_tokens = part_tokens
                else:
                    current_chunk.append(part)
                    current_chunk_tokens += part_tokens
        else:
            if current_chunk_tokens + tokens > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [comment]
                current_chunk_tokens = tokens
            else:
                current_chunk.append(comment)
                current_chunk_tokens += tokens
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def summarize_text_batch(texts, tokenizer, model, device, max_length=75, min_length=30):
    """Summarize a batch of texts using the provided model and tokenizer."""
    try:
        inputs = tokenizer(texts, truncation=True, padding=True, max_length=1024, return_tensors='pt').to(device)
        summary_ids = model.generate(inputs['input_ids'], max_length=max_length, min_length=min_length, num_beams=4)
        return tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
    except Exception as e:
        st.error(f"Error in summarization: {e}")
        return ["Error"] * len(texts)

def preprocess_comments_and_summarize(feedback_data, comment_column, batch_size=32, max_length=75, min_length=30, max_tokens=1000, summarize_threshold=30):
    """Preprocess and summarize comments, skipping summarization for short comments."""
    if comment_column not in feedback_data.columns:
        st.error(f"Comment column '{comment_column}' not found in CSV.")
        return {}
    model, tokenizer, device = get_summarization_model_and_tokenizer()
    feedback_data['preprocessed_comments'] = feedback_data[comment_column].apply(preprocess_text)
    comments = feedback_data['preprocessed_comments'].tolist()
    token_counts = [get_token_count(c, tokenizer) for c in comments]
    summaries_dict = {}
    
    print(f"Number of comments not summarized (token count <= {summarize_threshold}): {sum(1 for tc in token_counts if tc <= summarize_threshold)}")
    print(f"Number of comments to summarize: {sum(1 for tc in token_counts if tc > summarize_threshold)}")
    
    for comment, token_count in zip(comments, token_counts):
        if token_count <= summarize_threshold:
            summaries_dict[comment] = comment  # Skip summarization
        else:
            summary = summarize_text_batch([comment], tokenizer, model, device, max_length, min_length)[0]
            summaries_dict[comment] = summary
    return summaries_dict

def compute_keyword_embeddings(categories, model):
    """Compute embeddings for category keywords."""
    keyword_embeddings = {}
    for category, subcategories in categories.items():
        for subcategory, keywords in subcategories.items():
            for keyword in keywords:
                key = (category, subcategory, keyword)
                if key not in keyword_embeddings:
                    keyword_embeddings[key] = model.encode(keyword, show_progress_bar=False)
    return keyword_embeddings

def categorize_comments(feedback_data, categories, similarity_threshold, emerging_issue_mode, model):
    """Categorize comments based on similarity to keywords."""
    keyword_embeddings = compute_keyword_embeddings(categories, model)
    keyword_matrix = np.array(list(keyword_embeddings.values()))
    keyword_mapping = list(keyword_embeddings.keys())
    batch_size = 1024
    comment_embeddings = []
    comments = feedback_data['summarized_comments'].tolist()
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i + batch_size]
        comment_embeddings.extend(model.encode(batch, show_progress_bar=False))
    comment_matrix = np.array(comment_embeddings)
    similarity_matrix = cosine_similarity(comment_matrix, keyword_matrix)
    max_scores = similarity_matrix.max(axis=1)
    max_indices = similarity_matrix.argmax(axis=1)
    categories_list, subcats_list, keyphrases_list = [], [], []
    for score, idx in zip(max_scores, max_indices):
        cat, subcat, kw = keyword_mapping[idx]
        if emerging_issue_mode and score < similarity_threshold:
            cat, subcat = 'No Match', 'No Match'
        categories_list.append(cat)
        subcats_list.append(subcat)
        keyphrases_list.append(kw)
    feedback_data['Category'] = categories_list
    feedback_data['Sub-Category'] = subcats_list
    feedback_data['Keyphrase'] = keyphrases_list
    feedback_data['Best Match Score'] = max_scores
    return feedback_data

@st.cache_data(persist="disk", hash_funcs={dict: lambda x: str(sorted(x.items()))})
def process_feedback_data(feedback_data, comment_column, date_column, categories, similarity_threshold, emerging_issue_mode, summary_max_length, summary_min_length, summarize_threshold):
    """Process feedback data with summarization, categorization, and sentiment analysis."""
    if comment_column not in feedback_data.columns or date_column not in feedback_data.columns:
        st.error(f"Missing required column(s): '{comment_column}' or '{date_column}' not in CSV.")
        return pd.DataFrame()
    model = initialize_bert_model()
    sentiment_model = get_sentiment_model()
    summaries_dict = preprocess_comments_and_summarize(feedback_data, comment_column, max_length=summary_max_length, min_length=summary_min_length, summarize_threshold=summarize_threshold)
    feedback_data['preprocessed_comments'] = feedback_data[comment_column].apply(preprocess_text)
    feedback_data['summarized_comments'] = feedback_data['preprocessed_comments'].map(summaries_dict)
    feedback_data['Sentiment'] = feedback_data['preprocessed_comments'].apply(lambda x: perform_sentiment_analysis(x, sentiment_model))
    feedback_data = categorize_comments(feedback_data, categories, similarity_threshold, emerging_issue_mode, model)
    feedback_data['Parsed Date'] = pd.to_datetime(feedback_data[date_column], errors='coerce')
    feedback_data['Hour'] = feedback_data['Parsed Date'].dt.hour
    return feedback_data

# --- Streamlit Application ---
def main():
    st.set_page_config(layout="wide", page_title="Transcript Analysis Dashboard")
    st.title("📊 Transcript Categorization and Analysis Dashboard")
    st.markdown("Analyze feedback data with categorization, sentiment analysis, and summarization.")

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv", help="Upload a CSV file with feedback data.")
    similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.35, help="Threshold for keyword matching.")
    emerging_issue_mode = st.sidebar.checkbox("Enable Emerging Issue Detection", value=True, help="Cluster uncategorized comments.")
    chunk_size = st.sidebar.number_input("Chunk Size", min_value=32, value=32, step=32, help="Number of rows to process per batch.")
    
    # Summarization settings
    st.sidebar.header("Summarization Settings")
    summary_max_length = st.sidebar.number_input("Summary Max Length", min_value=10, value=75, step=5, help="Maximum length of the summary in tokens.")
    summary_min_length = st.sidebar.number_input("Summary Min Length", min_value=5, value=30, step=5, help="Minimum length of the summary in tokens.")
    summarize_threshold = st.sidebar.number_input("Summarize Threshold (tokens)", min_value=10, value=30, step=5, help="Comments with token count <= this value will not be summarized.")

    # Edit categories in sidebar
    st.sidebar.header("📋 Edit Categories")
    categories = default_categories.copy()
    new_categories = {}
    for category, subcategories in categories.items():
        category_name = st.sidebar.text_input(f"{category} Category", value=category)
        new_subcategories = {}
        for subcategory, keywords in subcategories.items():
            subcategory_name = st.sidebar.text_input(f"{subcategory} Subcategory under {category_name}", value=subcategory)
            with st.sidebar.expander(f"Keywords for {subcategory_name}"):
                category_keywords = st.text_area("Keywords", value="\n".join(keywords), help="Enter one keyword per line.")
            new_subcategories[subcategory_name] = [kw.strip() for kw in category_keywords.split("\n") if kw.strip()]
        new_categories[category_name] = new_subcategories
    categories = new_categories

    if uploaded_file:
        csv_data = uploaded_file.read()
        encoding = chardet.detect(csv_data)['encoding']
        uploaded_file.seek(0)
        total_rows = sum(1 for _ in uploaded_file) - 1  # Exclude header
        uploaded_file.seek(0)
        total_chunks = math.ceil(total_rows / chunk_size)
        
        try:
            first_chunk = next(pd.read_csv(BytesIO(csv_data), encoding=encoding, chunksize=1))
            column_names = first_chunk.columns.tolist()
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return
        
        comment_column = st.selectbox("Select Comment Column", column_names, help="Column with feedback text.")
        date_column = st.selectbox("Select Date Column", column_names, help="Column with date information.")
        grouping_option = st.radio("Group By", ["Date", "Week", "Month", "Quarter", "Hour"], help="Group trends by time period.")
        
        if st.button("Process Feedback", help="Start processing the uploaded feedback data."):
            chunk_iter = pd.read_csv(BytesIO(csv_data), encoding=encoding, chunksize=chunk_size)
            
            # Processing progress section
            st.header("Processing Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()
            category_counts_placeholder = st.empty()
            
            processed_chunks = []
            start_time = time.time()
            
            for i, chunk in enumerate(chunk_iter):
                status_text.text(f"Processing chunk {i+1}/{total_chunks}...")
                processed_chunk = process_feedback_data(
                    chunk, comment_column, date_column, categories,
                    similarity_threshold, emerging_issue_mode, summary_max_length, summary_min_length, summarize_threshold
                )
                if not processed_chunk.empty:
                    processed_chunks.append(processed_chunk)
                trends_data = pd.concat(processed_chunks, ignore_index=True)
                category_counts = trends_data['Category'].value_counts()
                category_counts_placeholder.bar_chart(category_counts)
                progress_bar.progress((i + 1) / total_chunks)
                eta = ((time.time() - start_time) / (i + 1)) * (total_chunks - (i + 1)) if i + 1 < total_chunks else 0
                status_text.text(f"Chunk {i+1}/{total_chunks} done. ETA: {int(eta)}s")
            
            if processed_chunks:
                trends_data = pd.concat(processed_chunks, ignore_index=True)
                trends_data = trends_data.drop_duplicates(subset=['preprocessed_comments', 'Parsed Date'])
                trends_data['Category'] = trends_data['Category'].str.strip().str.lower()
                trends_data['Sub-Category'] = trends_data['Sub-Category'].str.strip().str.lower()
                trends_data = trends_data.dropna(subset=['Category', 'Sub-Category', 'Sentiment'])
                
                # Complete analysis section
                st.header("Complete Analysis")
                st.subheader("📋 Processed Feedback Data")
                st.dataframe(trends_data)
                
                # Trends chart
                freq_map = {'Date': 'D', 'Week': 'W', 'Month': 'M', 'Quarter': 'Q', 'Hour': 'H'}
                if grouping_option != 'Hour':
                    trends_data_valid = trends_data.dropna(subset=['Parsed Date'])
                    if not trends_data_valid.empty:
                        trends_data_valid['Grouped Date'] = trends_data_valid['Parsed Date'].dt.to_period(freq_map[grouping_option]).dt.to_timestamp()
                        pivot_trends = trends_data_valid.groupby(['Grouped Date', 'Sub-Category']).size().unstack(fill_value=0)
                        top_subcats = pivot_trends.sum().nlargest(5).index
                        pivot_trends_top = pivot_trends[top_subcats]
                        st.subheader("📈 Top 5 Sub-Category Trends Over Time")
                        st.line_chart(pivot_trends_top)
                    else:
                        st.subheader("📈 Top 5 Sub-Category Trends Over Time")
                        st.warning("No data available for trends chart.")
                else:
                    pivot_trends = trends_data.groupby(['Hour', 'Sub-Category']).size().unstack(fill_value=0)
                    top_subcats = pivot_trends.sum().nlargest(5).index
                    pivot_trends_top = pivot_trends[top_subcats]
                    st.subheader("📈 Top 5 Sub-Category Trends by Hour")
                    st.line_chart(pivot_trends_top)

if __name__ == "__main__":
    main()
