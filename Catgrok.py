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

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "true"
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

@st.cache_resource
def get_summarizer():
    """Initialize and cache the summarizer pipeline."""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)

# --- Utility Functions ---
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.replace('\n', ' ').replace('\r', '').replace(' ', ' ')
    return re.sub(r'\s+', ' ', text).strip()

def perform_sentiment_analysis(text, sentiment_model=None, use_nltk=True):
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
    try:
        return len(tokenizer.encode(text)) - 2
    except Exception:
        return 0

def split_comments_into_chunks(comments, tokenizer, max_tokens=1000):
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
    try:
        inputs = tokenizer(texts, truncation=True, padding=True, max_length=1024, return_tensors='pt').to(device)
        summary_ids = model.generate(inputs['input_ids'], max_length=max_length, min_length=min_length, num_beams=4)
        return tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
    except Exception as e:
        st.error(f"Error in summarization: {e}")
        return ["Error"] * len(texts)

def preprocess_comments_and_summarize(feedback_data, comment_column, batch_size=32, max_length=75, min_length=30, max_tokens=1000, very_short_limit=30):
    if comment_column not in feedback_data.columns:
        st.error(f"Comment column '{comment_column}' not found in CSV.")
        return feedback_data
    model, tokenizer, device = get_summarization_model_and_tokenizer()
    feedback_data['preprocessed_comments'] = feedback_data[comment_column].apply(preprocess_text)
    comments = feedback_data['preprocessed_comments'].tolist()
    very_short = [(c, get_token_count(c, tokenizer)) for c in comments if get_token_count(c, tokenizer) <= very_short_limit]
    short = [(c, get_token_count(c, tokenizer)) for c in comments if very_short_limit < get_token_count(c, tokenizer) <= max_tokens]
    long = [(c, get_token_count(c, tokenizer)) for c in comments if get_token_count(c, tokenizer) > max_tokens]
    summaries_dict = {c: c for c, _ in very_short}
    short_texts = [c for c, _ in short]
    for i in tqdm(range(0, len(short_texts), batch_size), desc="Summarizing short comments"):
        batch = short_texts[i:i + batch_size]
        summaries = summarize_text_batch(batch, tokenizer, model, device, max_length, min_length)
        for orig, summ in zip(batch, summaries):
            summaries_dict[orig] = summ
    for comment, tokens in tqdm(long, desc="Summarizing long comments"):
        chunks = split_comments_into_chunks([(comment, tokens)], tokenizer, max_tokens)
        chunk_summaries = summarize_text_batch(chunks, tokenizer, model, device, max_length, min_length)
        full_summary = " ".join(chunk_summaries)
        while get_token_count(full_summary, tokenizer) > max_length:
            full_summary = summarize_text_batch([full_summary], tokenizer, model, device, max_length, min_length)[0]
        summaries_dict[comment] = full_summary
    feedback_data['summarized_comments'] = feedback_data['preprocessed_comments'].map(summaries_dict)
    return feedback_data

def compute_keyword_embeddings(categories, model):
    keyword_embeddings = {}
    for category, subcategories in categories.items():
        for subcategory, keywords in subcategories.items():
            for keyword in keywords:
                key = (category, subcategory, keyword)
                if key not in keyword_embeddings:
                    keyword_embeddings[key] = model.encode(keyword, show_progress_bar=False)
    return keyword_embeddings

def categorize_comments(feedback_data, categories, similarity_threshold, emerging_issue_mode, model):
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
    feedback_data['embeddings'] = list(comment_matrix)
    return feedback_data

def summarize_cluster(comments, summarizer):
    combined_text = " ".join(comments)
    try:
        summary = summarizer(combined_text, max_length=15, min_length=5, do_sample=False)[0]['summary_text']
        return summary.strip().capitalize()
    except Exception:
        return "Unnamed Cluster"

def cluster_no_match_comments(feedback_data, summarizer, max_clusters=10):
    no_match_idx = feedback_data['Category'] == 'No Match'
    if no_match_idx.sum() < 2:
        st.warning("Not enough 'No Match' comments to cluster.")
        return feedback_data
    no_match_embeddings = np.array(feedback_data.loc[no_match_idx, 'embeddings'].tolist())
    no_match_embeddings = normalize(no_match_embeddings)
    k = min(math.ceil(math.sqrt(no_match_idx.sum())), max_clusters)
    try:
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(no_match_embeddings)
    except Exception as e:
        st.error(f"Error in clustering: {e}")
        return feedback_data
    cluster_comments = defaultdict(list)
    for i, idx in enumerate(feedback_data.index[no_match_idx]):
        cluster_comments[clusters[i]].append(feedback_data.at[idx, 'preprocessed_comments'])
    cluster_summaries = {}
    for cid, comments in cluster_comments.items():
        cluster_summaries[cid] = summarize_cluster(comments, summarizer)
        print(f"Cluster {cid} summary: {cluster_summaries[cid]}")
    for i, idx in enumerate(feedback_data.index[no_match_idx]):
        cluster_id = clusters[i]
        feedback_data.at[idx, 'Category'] = 'Emerging Issues'
        feedback_data.at[idx, 'Sub-Category'] = cluster_summaries[cluster_id]
    return feedback_data

@st.cache_data(persist="disk", hash_funcs={dict: lambda x: str(sorted(x.items()))})
def process_feedback_data(feedback_data, comment_column, date_column, categories, similarity_threshold, emerging_issue_mode, max_clusters=10):
    if comment_column not in feedback_data.columns or date_column not in feedback_data.columns:
        st.error(f"Missing required column(s): '{comment_column}' or '{date_column}' not in CSV.")
        return pd.DataFrame()
    model = initialize_bert_model()
    sentiment_model = get_sentiment_model()
    summarizer = get_summarizer()
    feedback_data = preprocess_comments_and_summarize(feedback_data, comment_column)
    feedback_data['Sentiment'] = feedback_data['preprocessed_comments'].apply(lambda x: perform_sentiment_analysis(x, sentiment_model))
    feedback_data = categorize_comments(feedback_data, categories, similarity_threshold, emerging_issue_mode, model)
    if emerging_issue_mode:
        feedback_data = cluster_no_match_comments(feedback_data, summarizer, max_clusters)
    feedback_data['Parsed Date'] = pd.to_datetime(feedback_data[date_column], errors='coerce')
    feedback_data['Hour'] = feedback_data['Parsed Date'].dt.hour
    if 'embeddings' in feedback_data.columns:
        feedback_data.drop(columns=['embeddings'], inplace=True)
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
    emerging_issue_mode = st.sidebar.checkbox("Enable Emerging Issue Detection", value=True, help="Cluster uncategorized comments and name them with summaries.")
    chunk_size = st.sidebar.number_input("Chunk Size", min_value=32, value=32, step=32, help="Number of rows to process per batch.")
    max_clusters = st.sidebar.number_input("Maximum Clusters for Emerging Issues", min_value=1, max_value=50, value=10, help="Max clusters for uncategorized comments.")
    
    # Category editing in sidebar
    st.sidebar.header("📋 Edit Categories")
    categories = default_categories.copy()  # Avoid modifying the original
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
        total_rows = sum(1 for _ in uploaded_file) - 1
        uploaded_file.seek(0)
        total_chunks = math.ceil(total_rows / chunk_size) if total_rows > 0 else 1
        
        try:
            first_chunk = next(pd.read_csv(BytesIO(csv_data), encoding=encoding, chunksize=1))
            column_names = first_chunk.columns.tolist()
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return
        
        comment_column = st.selectbox("Select Comment Column", column_names, help="Column containing feedback text.")
        date_column = st.selectbox("Select Date Column", column_names, help="Column containing date information.")
        grouping_option = st.radio("Group By", ["Date", "Week", "Month", "Quarter", "Hour"], help="Group trends by time period.")
        
        # Initialize placeholders for UI components
        processed_data_placeholder = st.empty()
        trends_chart_placeholder = st.empty()
        sentiment_dist_placeholder = st.empty()  # New: Sentiment distribution plot
        keyword_freq_placeholder = st.empty()    # New: Keyword frequency plot
        category_sentiment_df_placeholder = st.empty()  # New: Average sentiment per category
        subcategory_sentiment_df_placeholder = st.empty()  # New: Average sentiment per subcategory
        top_comments_placeholder = st.empty()
        emerging_issues_placeholder = st.empty()  # New: Emerging issues widgets
        download_placeholder = st.empty()

        if st.button("Process Feedback", help="Start processing the uploaded feedback data."):
            chunk_iter = pd.read_csv(BytesIO(csv_data), encoding=encoding, chunksize=chunk_size)
            progress_bar = st.progress(0)
            status_text = st.empty()
            trends_data_list = []
            start_time = time.time()
            
            for i, chunk in enumerate(chunk_iter):
                status_text.text(f"Processing chunk {i+1}/{total_chunks}...")
                processed_chunk = process_feedback_data(chunk, comment_column, date_column, categories, 
                                                       similarity_threshold, emerging_issue_mode, max_clusters)
                if not processed_chunk.empty:
                    trends_data_list.append(processed_chunk)
                progress_bar.progress((i + 1) / total_chunks)
                eta = ((time.time() - start_time) / (i + 1)) * (total_chunks - (i + 1)) if i + 1 < total_chunks else 0
                status_text.text(f"Chunk {i+1}/{total_chunks} done. ETA: {int(eta)}s")
                
                # Concatenate cumulative data for real-time updates
                if trends_data_list:
                    trends_data = pd.concat(trends_data_list, ignore_index=True)
                    
                    # Remove duplicates to prevent inflated counts
                    trends_data = trends_data.drop_duplicates(subset=['preprocessed_comments', 'Parsed Date'])
                    
                    # Normalize category and subcategory names
                    trends_data['Category'] = trends_data['Category'].str.strip().str.lower()
                    trends_data['Sub-Category'] = trends_data['Sub-Category'].str.strip().str.lower()
                    
                    # Ensure no NaN values in critical columns
                    trends_data = trends_data.dropna(subset=['Category', 'Sub-Category', 'Sentiment'])
                    
                    # Processed Feedback Data with Sentiment Colors
                    def color_sentiment(val):
                        color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
                        return f'color: {color}'
                    styled_trends_data = trends_data.style.applymap(color_sentiment, subset=['Sentiment'])
                    with processed_data_placeholder:
                        st.subheader("📋 Processed Feedback Data")
                        st.dataframe(styled_trends_data, use_container_width=True)
                    
                    # Trends Chart - Using Streamlit's line chart
                    freq_map = {'Date': 'D', 'Week': 'W', 'Month': 'M', 'Quarter': 'Q', 'Hour': 'H'}
                    if grouping_option != 'Hour':
                        trends_data_valid = trends_data.dropna(subset=['Parsed Date'])
                        if not trends_data_valid.empty:
                            trends_data_valid['Grouped Date'] = trends_data_valid['Parsed Date'].dt.to_period(freq_map[grouping_option]).dt.to_timestamp()
                            pivot_trends = trends_data_valid.groupby(['Grouped Date', 'Sub-Category']).size().unstack(fill_value=0)
                            top_subcats = pivot_trends.sum().nlargest(5).index
                            pivot_trends_top = pivot_trends[top_subcats]
                            with trends_chart_placeholder:
                                st.subheader("📈 Top 5 Sub-Category Trends Over Time")
                                st.line_chart(pivot_trends_top)
                        else:
                            with trends_chart_placeholder:
                                st.subheader("📈 Top 5 Sub-Category Trends Over Time")
                                st.warning("No data available for trends chart.")
                    else:
                        pivot_trends = trends_data.groupby(['Hour', 'Sub-Category']).size().unstack(fill_value=0)
                        top_subcats = pivot_trends.sum().nlargest(5).index
                        pivot_trends_top = pivot_trends[top_subcats]
                        with trends_chart_placeholder:
                            st.subheader("📈 Top 5 Sub-Category Trends by Hour")
                            st.line_chart(pivot_trends_top)

                    # --- Sentiment Distribution Plot ---
                    with sentiment_dist_placeholder:
                        st.subheader("📊 Sentiment Distribution")
                        sentiment_bins = pd.cut(trends_data['Sentiment'], bins=[-1, -0.5, 0.5, 1], labels=['Negative', 'Neutral', 'Positive'])
                        sentiment_dist = sentiment_bins.value_counts().sort_index()
                        st.bar_chart(sentiment_dist)
                    
                    # --- Keyword Frequency Plot ---
                    with keyword_freq_placeholder:
                        st.subheader("🔑 Top 10 Keywords by Frequency")
                        keyword_counts = trends_data['Keyphrase'].value_counts().head(10)
                        if not keyword_counts.empty:
                            st.bar_chart(keyword_counts)
                        else:
                            st.warning("No keyword data available.")

                    # --- Average Sentiment per Category DataFrame ---
                    with category_sentiment_df_placeholder:
                        st.subheader("📊 Average Sentiment per Category")
                        avg_sentiment_category = trends_data.groupby('Category')['Sentiment'].mean().sort_values(ascending=False)
                        st.dataframe(avg_sentiment_category)

                    # --- Average Sentiment per Sub-Category DataFrame ---
                    with subcategory_sentiment_df_placeholder:
                        st.subheader("📊 Average Sentiment per Sub-Category")
                        avg_sentiment_subcategory = trends_data.groupby(['Category', 'Sub-Category'])['Sentiment'].mean().sort_values(ascending=False)
                        st.dataframe(avg_sentiment_subcategory)

                    # Top 10 Recent Comments by Sub-Category with Enhanced Columns
                    with top_comments_placeholder:
                        st.subheader("📝 Top 10 Recent Comments by Sub-Category")
                        pivot_subcat_counts = trends_data.groupby(['Category', 'Sub-Category']).size().sort_values(ascending=False)
                        if not pivot_subcat_counts.empty:
                            top_subcats = pivot_subcat_counts.head(10).index.get_level_values('Sub-Category')
                            for subcat in top_subcats:
                                with st.expander(f"Comments for {subcat}"):
                                    st.markdown(f"### {subcat}")
                                    filtered = trends_data[trends_data['Sub-Category'] == subcat].nlargest(10, 'Parsed Date')
                                    st.table(filtered[['Category', 'Parsed Date', comment_column, 'summarized_comments', 'Sentiment']])
                                    st.markdown("---")
                        else:
                            st.write("No sub-category data available.")

                    # --- Emerging Issues Widgets (if enabled) ---
                    if emerging_issue_mode:
                        with emerging_issues_placeholder:
                            st.subheader("🔍 Emerging Issues")
                            emerging_issues = trends_data[trends_data['Category'] == 'emerging issues']
                            if not emerging_issues.empty:
                                cluster_names = emerging_issues['Sub-Category'].unique()
                                selected_cluster = st.selectbox("Select an Emerging Issue Cluster", cluster_names)
                                cluster_data = emerging_issues[emerging_issues['Sub-Category'] == selected_cluster]
                                st.write(f"Comments in cluster '{selected_cluster}':")
                                st.table(cluster_data[['Parsed Date', comment_column, 'summarized_comments', 'Sentiment']])
                            else:
                                st.write("No emerging issues detected.")

            # After processing, provide Excel download
            if trends_data_list:
                trends_data = pd.concat(trends_data_list, ignore_index=True)
                trends_data = trends_data.drop_duplicates(subset=['preprocessed_comments', 'Parsed Date'])
                trends_data = trends_data.dropna(subset=['Category', 'Sub-Category', 'Sentiment'])
                excel_file = BytesIO()
                with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                    trends_data.to_excel(writer, sheet_name='Feedback Trends', index=False)
                    if 'pivot_trends' in locals():
                        pivot_trends.to_excel(writer, sheet_name=f'Trends by {grouping_option}')
                    if not pivot_subcat_counts.empty:
                        comments_sheet = writer.book.add_worksheet('Example Comments')
                        start_row = 0
                        for subcat in top_subcats:
                            filtered = trends_data[trends_data['Sub-Category'] == subcat].nlargest(10, 'Parsed Date')
                            comments_sheet.merge_range(start_row, 0, start_row, 1, subcat)
                            comments_sheet.write(start_row + 1, 0, 'Date')
                            comments_sheet.write(start_row + 1, 1, comment_column)
                            for i, (_, row) in enumerate(filtered.iterrows(), start=start_row + 2):
                                comments_sheet.write(i, 0, row['Parsed Date'].strftime('%Y-%m-%d') if pd.notna(row['Parsed Date']) else '')
                                comments_sheet.write_string(i, 1, str(row[comment_column]))
                            start_row += 12
                excel_file.seek(0)
                with download_placeholder:
                    st.download_button("Download Excel", data=excel_file, file_name="feedback_trends.xlsx", 
                                      mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                      help="Download the processed data as an Excel file.")
            else:
                st.error("No valid data processed for export.")

if __name__ == "__main__":
    main()
