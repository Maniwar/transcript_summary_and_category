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
import plotly.express as px
from datetime import datetime
from collections import defaultdict

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "true"
nltk.download('vader_lexicon', quiet=True)

# --- Custom Dataset for Summarization ---
class SummarizationDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze()

# --- Utility Functions ---
def preprocess_text(text):
    """Preprocess text by removing special characters, normalizing whitespace, and converting to ASCII."""
    if pd.isna(text):
        return ""
    text = str(text).encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.replace('\n', ' ').replace('\r', '').replace(' ', ' ')
    return re.sub(r'\s+', ' ', text).strip()

# --- Model Initialization ---
@st.cache_resource
def initialize_bert_model():
    """Initialize and cache the BERT model for embedding generation."""
    start_time = time.time()
    print("Initializing BERT model...")
    model = SentenceTransformer('all-mpnet-base-v2', device='cpu')
    print(f"BERT model initialized. Time taken: {time.time() - start_time} seconds.")
    return model

@st.cache_resource
def get_summarization_model_and_tokenizer():
    """Initialize and cache the summarization model and tokenizer."""
    model_name = "knkarthick/MEETING_SUMMARY"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return model, tokenizer, device

@st.cache_resource
def get_sentiment_model():
    """Initialize and cache the sentiment analysis model."""
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=0 if torch.cuda.is_available() else -1)

@st.cache_resource
def get_summarizer():
    """Initialize and cache the summarizer pipeline."""
    return pipeline("summarization", model="facebook/bart-large-cnn",
                    device=0 if torch.cuda.is_available() else -1)

# --- Processing Functions ---
def perform_sentiment_analysis(text, sentiment_model=None, use_nltk=True):
    """Perform sentiment analysis using NLTK or a transformer model."""
    if not isinstance(text, str):
        return 0.0  # Handle non-string inputs gracefully
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
    """Count the number of tokens in the text using the tokenizer."""
    try:
        return len(tokenizer.encode(text)) - 2
    except Exception:
        return 0  # Handle invalid text gracefully

def split_comments_into_chunks(comments, tokenizer, max_tokens=1000):
    """Split long comments into chunks based on token count."""
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
    """Summarize a batch of texts using the summarization model."""
    try:
        inputs = tokenizer(texts, truncation=True, padding=True, max_length=1024, return_tensors='pt').to(device)
        summary_ids = model.generate(inputs['input_ids'], max_length=max_length, min_length=min_length, num_beams=4)
        return tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
    except Exception as e:
        st.error(f"Error in summarization: {e}")
        return ["Error"] * len(texts)

def preprocess_comments_and_summarize(feedback_data, comment_column, batch_size=32, max_length=75, min_length=30, max_tokens=1000, very_short_limit=30):
    """Preprocess and summarize comments, handling very short, short, and long comments separately."""
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
    """Compute embeddings for all keywords in the categories."""
    keyword_embeddings = {}
    for category, subcategories in categories.items():
        for subcategory, keywords in subcategories.items():
            for keyword in keywords:
                key = (category, subcategory, keyword)
                if key not in keyword_embeddings:
                    keyword_embeddings[key] = model.encode(keyword, show_progress_bar=False)
    return keyword_embeddings

def categorize_comments(feedback_data, categories, similarity_threshold, emerging_issue_mode, model):
    """Categorize comments based on similarity to category keywords."""
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
    """Summarize a cluster of comments to generate a representative name."""
    combined_text = " ".join(comments)
    try:
        summary = summarizer(combined_text, max_length=15, min_length=5, do_sample=False)[0]['summary_text']
        return summary.strip().capitalize()
    except Exception:
        return "Unnamed Cluster"

def cluster_no_match_comments(feedback_data, summarizer):
    """Cluster 'No Match' comments and assign them to new 'Emerging Issues' categories."""
    no_match_idx = feedback_data['Category'] == 'No Match'
    if no_match_idx.sum() <= 10:
        return feedback_data
    
    no_match_embeddings = np.array(feedback_data.loc[no_match_idx, 'embeddings'].tolist())
    no_match_embeddings = normalize(no_match_embeddings)
    k = min(math.ceil(math.sqrt(no_match_idx.sum())), 10)
    try:
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(no_match_embeddings)
    except Exception as e:
        st.error(f"Error in clustering: {e}")
        return feedback_data
    
    cluster_comments = defaultdict(list)
    for i, idx in enumerate(feedback_data.index[no_match_idx]):
        cluster_comments[clusters[i]].append(feedback_data.at[idx, 'preprocessed_comments'])
    
    cluster_summaries = {cid: summarize_cluster(comments, summarizer) for cid, comments in cluster_comments.items()}
    
    for i, idx in enumerate(feedback_data.index[no_match_idx]):
        cluster_id = clusters[i]
        feedback_data.at[idx, 'Category'] = 'Emerging Issues'
        feedback_data.at[idx, 'Sub-Category'] = cluster_summaries[cluster_id]
    
    return feedback_data

@st.cache_data(persist="disk", hash_funcs={dict: lambda x: str(sorted(x.items()))})
def process_feedback_data(feedback_data, comment_column, date_column, categories, similarity_threshold, emerging_issue_mode):
    """Main function to process feedback data: summarize, categorize, analyze sentiment, and cluster."""
    if comment_column not in feedback_data.columns or date_column not in feedback_data.columns:
        st.error(f"Missing required column(s): '{comment_column}' or '{date_column}' not in CSV.")
        return pd.DataFrame()
    
    model = initialize_bert_model()
    sentiment_model = get_sentiment_model()
    summarizer = get_summarizer()
    
    feedback_data = preprocess_comments_and_summarize(feedback_data, comment_column)
    feedback_data['Sentiment'] = feedback_data['preprocessed_comments'].apply(
        lambda x: perform_sentiment_analysis(x, sentiment_model))
    feedback_data = categorize_comments(feedback_data, categories, similarity_threshold, emerging_issue_mode, model)
    if emerging_issue_mode:
        feedback_data = cluster_no_match_comments(feedback_data, summarizer)
    feedback_data['Parsed Date'] = pd.to_datetime(feedback_data[date_column], errors='coerce')
    feedback_data['Hour'] = feedback_data['Parsed Date'].dt.hour
    if 'embeddings' in feedback_data.columns:
        feedback_data.drop(columns=['embeddings'], inplace=True)
    return feedback_data

# --- Streamlit Application ---
def main():
    st.set_page_config(layout="wide")
    st.title("👨‍💻 Transcript Categorization and Analysis")

    # Sidebar configuration
    st.sidebar.header("Configuration")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
    similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.35)
    emerging_issue_mode = st.sidebar.checkbox("Enable Emerging Issue Detection", value=True)
    st.sidebar.write("Emerging issue mode clusters uncategorized comments and names them with summaries.")
    chunk_size = st.sidebar.number_input("Chunk Size", min_value=32, value=32, step=32)
    
    # Category editing in sidebar
    st.sidebar.header("Edit Categories")
    categories = default_categories.copy()  # Avoid modifying the original
    new_categories = {}
    for category, subcategories in categories.items():
        category_name = st.sidebar.text_input(f"{category} Category", value=category)
        new_subcategories = {}
        for subcategory, keywords in subcategories.items():
            subcategory_name = st.sidebar.text_input(f"{subcategory} Subcategory under {category_name}", value=subcategory)
            with st.sidebar.expander(f"Keywords for {subcategory_name}"):
                category_keywords = st.text_area("Keywords", value="\n".join(keywords))
            new_subcategories[subcategory_name] = category_keywords.split("\n")
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
        
        comment_column = st.selectbox("Select comment column", column_names)
        date_column = st.selectbox("Select date column", column_names)
        grouping_option = st.radio("Group by", ["Date", "Week", "Month", "Quarter", "Hour"])
        
        if st.button("Process Feedback"):
            chunk_iter = pd.read_csv(BytesIO(csv_data), encoding=encoding, chunksize=chunk_size)
            progress_bar = st.progress(0)
            status_text = st.empty()
            trends_data_list = []
            start_time = time.time()
            
            for i, chunk in enumerate(chunk_iter):
                status_text.text(f"Processing chunk {i+1}/{total_chunks}...")
                processed_chunk = process_feedback_data(chunk, comment_column, date_column, categories, 
                                                       similarity_threshold, emerging_issue_mode)
                if not processed_chunk.empty:
                    trends_data_list.append(processed_chunk)
                progress_bar.progress((i + 1) / total_chunks)
                eta = ((time.time() - start_time) / (i + 1)) * (total_chunks - (i + 1)) if i + 1 < total_chunks else 0
                status_text.text(f"Chunk {i+1}/{total_chunks} done. ETA: {int(eta)}s")
                
                # Concatenate cumulative data for real-time updates
                if trends_data_list:
                    trends_data = pd.concat(trends_data_list, ignore_index=True)
                    
                    # Display processed data
                    st.subheader("Processed Feedback Data")
                    st.dataframe(trends_data)
                    
                    # All Categories Trends
                    st.subheader("All Categories Trends")
                    freq_map = {'Date': 'D', 'Week': 'W-SUN', 'Month': 'M', 'Quarter': 'Q', 'Hour': 'H'}
                    if grouping_option != 'Hour':
                        # Filter out rows with NaT in 'Parsed Date' to avoid grouping issues
                        trends_data_valid = trends_data.dropna(subset=['Parsed Date'])
                        if not trends_data_valid.empty:
                            pivot = trends_data_valid.pivot_table(
                                index=['Category', 'Sub-Category'],
                                columns=pd.Grouper(key='Parsed Date', freq=freq_map[grouping_option]),
                                values='Sentiment',
                                aggfunc='count',
                                fill_value=0
                            ).reset_index()
                        else:
                            st.warning("No valid dates for trend analysis.")
                            pivot = pd.DataFrame()
                    else:
                        pivot = trends_data.pivot_table(
                            index=['Category', 'Sub-Category'],
                            columns='Hour',
                            values='Sentiment',
                            aggfunc='count',
                            fill_value=0
                        ).reset_index()

                    if not pivot.empty:
                        # Melt the pivot table to long format for plotting
                        melted = pd.melt(
                            pivot,
                            id_vars=['Category', 'Sub-Category'],
                            var_name='Date' if grouping_option != 'Hour' else 'Hour',
                            value_name='Sentiment Count'
                        )

                        # Filter for top 5 sub-categories based on total sentiment count
                        numeric_cols = pivot.select_dtypes(include=[np.number]).columns
                        top_subcats = pivot.groupby('Sub-Category')[numeric_cols].sum().sum(axis=1).nlargest(5).index
                        melted_top = melted[melted['Sub-Category'].isin(top_subcats)]

                        # Plot the line chart
                        if grouping_option != 'Hour':
                            fig_trends = px.line(
                                melted_top,
                                x='Date',
                                y='Sentiment Count',
                                color='Sub-Category',
                                title="Top 5 Trends"
                            )
                        else:
                            fig_trends = px.line(
                                melted_top,
                                x='Hour',
                                y='Sentiment Count',
                                color='Sub-Category',
                                title="Top 5 Trends by Hour"
                            )
                        st.plotly_chart(fig_trends)
                    
                    # Category vs Sentiment and Quantity
                    st.subheader("Category vs Sentiment and Quantity")
                    pivot1 = trends_data.groupby('Category')['Sentiment'].agg(['mean', 'count']).sort_values('count', ascending=False)
                    st.dataframe(pivot1)
                    fig_cat = px.bar(pivot1, x=pivot1.index, y='count', title="Category Quantity")
                    st.plotly_chart(fig_cat)
                    
                    # Sub-Category vs Sentiment and Quantity
                    st.subheader("Sub-Category vs Sentiment and Quantity")
                    pivot2 = trends_data.groupby(['Category', 'Sub-Category'])['Sentiment'].agg(['mean', 'count']).sort_values('count', ascending=False)
                    st.dataframe(pivot2)
                    fig_subcat = px.bar(pivot2, x=pivot2.index.get_level_values('Sub-Category'), y='count', 
                                       color=pivot2.index.get_level_values('Category'), title="Sub-Category Quantity")
                    st.plotly_chart(fig_subcat)
                    
                    # Top 10 Recent Comments by Sub-Category
                    st.subheader("Top 10 Recent Comments by Sub-Category")
                    top_subcats = pivot2.head(10).index.get_level_values('Sub-Category')
                    for subcat in top_subcats:
                        st.write(f"**{subcat}**")
                        filtered = trends_data[trends_data['Sub-Category'] == subcat].nlargest(10, 'Parsed Date')
                        st.table(filtered[['Parsed Date', comment_column, 'summarized_comments', 'Sentiment']])
                else:
                    st.warning("No data processed yet.")
            
            # After all chunks are processed, provide Excel download
            if trends_data_list:
                trends_data = pd.concat(trends_data_list, ignore_index=True)
                excel_file = BytesIO()
                with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                    trends_data.to_excel(writer, sheet_name='Feedback Trends', index=False)
                    if not pivot.empty:
                        pivot.to_excel(writer, sheet_name=f'Trends by {grouping_option}')
                    pivot1.to_excel(writer, sheet_name='Categories')
                    pivot2.to_excel(writer, sheet_name='Subcategories')
                    
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
                b64 = base64.b64encode(excel_file.read()).decode()
                st.download_button("Download Excel", data=excel_file, file_name="feedback_trends.xlsx", 
                                  mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.error("No valid data processed for export.")

if __name__ == "__main__":
    main()
