import os
import time
import math
import json
import logging
import psutil
import shutil
import glob
import cProfile
import pstats
from io import BytesIO, StringIO
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import normalize
import chardet
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st
import textwrap
from categories_josh_sub_V6_3 import default_categories  # Assume this is a local file with category definitions
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import xlsxwriter
from wordcloud import WordCloud
from concurrent.futures import ThreadPoolExecutor

# Set environment variables for local operation
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Download required NLTK data locally (runs once during setup)
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Configure logging to a local file
logging.basicConfig(
    filename="transcript_app_local.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Constants and Configurations ---
SUPPORTED_LANGUAGES = {"en": "English"}  # Limited to English for simplicity
THEME_OPTIONS = {"Light": "light", "Dark": "dark", "Custom": "custom"}
LOCAL_STORAGE_DIR = "local_storage"
STOPWORDS = set(stopwords.words('english'))

# Ensure local storage directory exists
os.makedirs(LOCAL_STORAGE_DIR, exist_ok=True)

# --- Device Detection ---
def get_device() -> str:
    """Determine the best available device for computation."""
    if torch.cuda.is_available():
        logger.info("Using CUDA device.")
        return 'cuda'
    elif torch.backends.mps.is_available():
        logger.info("Using MPS device.")
        return 'mps'
    else:
        logger.info("Using CPU device.")
        return 'cpu'

# --- Model Initialization ---
@st.cache_resource
def initialize_bert_model() -> SentenceTransformer:
    """Initialize and cache the BERT model locally for embeddings."""
    device = get_device()
    model = SentenceTransformer('all-mpnet-base-v2', device=device)
    logger.info(f"BERT model initialized on {device}.")
    return model

@st.cache_resource
def get_summarization_model_and_tokenizer(model_name: str = "knkarthick/MEETING_SUMMARY") -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer, str]:
    """Initialize and cache the summarization model and tokenizer locally."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = get_device()
    model.to(device)
    logger.info(f"Summarization model {model_name} loaded on {device}.")
    return model, tokenizer, device

@st.cache_resource
def get_sentiment_model(model_name: str = "distilbert-base-uncased-finetuned-sst-2-english") -> pipeline:
    """Initialize and cache the sentiment analysis model locally."""
    device = 0 if torch.cuda.is_available() else -1
    model = pipeline("sentiment-analysis", model=model_name, device=device)
    logger.info(f"Sentiment model {model_name} initialized.")
    return model

# --- Utility Functions ---
def preprocess_text(text: str, custom_regex: Optional[str] = None) -> str:
    """Preprocess text by removing special characters and normalizing whitespace."""
    if pd.isna(text):
        return ""
    text = str(text).encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.replace('\n', ' ').replace('\r', '').replace(' ', ' ')
    if custom_regex:
        text = re.sub(custom_regex, '', text)
    processed_text = re.sub(r'\s+', ' ', text).strip()
    logger.debug(f"Preprocessed text: {processed_text[:50]}...")
    return processed_text

def save_local_file(content: bytes, filename: str, directory: str = LOCAL_STORAGE_DIR) -> str:
    """Save content to a local file."""
    filepath = os.path.join(directory, filename)
    with open(filepath, 'wb') as f:
        f.write(content)
    logger.info(f"Saved file: {filepath}")
    return filepath

def perform_sentiment_analysis(text: str, sentiment_model=None, use_nltk: bool = True) -> float:
    """Perform sentiment analysis using NLTK's VADER or a transformer model."""
    if not isinstance(text, str):
        return 0.0
    if use_nltk:
        analyzer = SentimentIntensityAnalyzer()
        score = analyzer.polarity_scores(text)['compound']
    else:
        try:
            result = sentiment_model(text[:512])[0]
            score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            score = 0.0
    logger.debug(f"Sentiment score for '{text[:50]}...': {score}")
    return score

def get_token_count(text: str, tokenizer: AutoTokenizer) -> int:
    """Count tokens in the text using the provided tokenizer."""
    try:
        count = len(tokenizer.encode(text)) - 2
        return count
    except Exception:
        return 0

def split_comments_into_chunks(comments: List[Tuple[str, int]], tokenizer: AutoTokenizer, max_tokens: int = 1000) -> List[str]:
    """Split comments into chunks based on token limits."""
    sorted_comments = sorted(comments, key=lambda x: x[1], reverse=True)
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0
    for comment, tokens in sorted_comments:
        if tokens > max_tokens:
            parts = textwrap.wrap(comment, width=int(max_tokens / 2))
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
    logger.info(f"Split {len(sorted_comments)} comments into {len(chunks)} chunks.")
    return chunks

def summarize_text_batch(texts: List[str], tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM, device: str, max_length: int = 75, min_length: int = 30, num_beams: int = 4) -> List[str]:
    """Summarize a batch of texts using the provided model."""
    try:
        inputs = tokenizer(texts, truncation=True, padding=True, max_length=1024, return_tensors='pt').to(device)
        summary_ids = model.generate(inputs['input_ids'], max_length=max_length, min_length=min_length, num_beams=num_beams)
        summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
        return summaries
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return ["Error"] * len(texts)

def preprocess_comments_and_summarize(feedback_data: pd.DataFrame, comment_column: str, model_name: str, batch_size: int = 32, 
                                     max_length: int = 75, min_length: int = 30, max_tokens: int = 1000, summarize_threshold: int = 30, 
                                     num_beams: int = 4, custom_regex: Optional[str] = None) -> Dict[str, str]:
    """Preprocess and summarize comments with parallel processing."""
    if comment_column not in feedback_data.columns:
        raise ValueError(f"Column '{comment_column}' not found in CSV.")
    model, tokenizer, device = get_summarization_model_and_tokenizer(model_name)
    feedback_data['preprocessed_comments'] = feedback_data[comment_column].apply(lambda x: preprocess_text(x, custom_regex))
    comments = feedback_data['preprocessed_comments'].tolist()
    token_counts = [get_token_count(c, tokenizer) for c in comments]
    summaries_dict = {}
    
    to_summarize = [(c, tc) for c, tc in zip(comments, token_counts) if tc > summarize_threshold]
    
    def summarize_batch(batch: List[Tuple[str, int]]) -> List[Tuple[str, str]]:
        texts = [x[0] for x in batch]
        summaries = summarize_text_batch(texts, tokenizer, model, device, max_length, min_length, num_beams)
        return list(zip(texts, summaries))
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        batches = [to_summarize[i:i + batch_size] for i in range(0, len(to_summarize), batch_size)]
        results = list(executor.map(summarize_batch, batches))
    
    for batch_result in results:
        for comment, summary in batch_result:
            summaries_dict[comment] = summary
    for comment, tc in zip(comments, token_counts):
        if tc <= summarize_threshold:
            summaries_dict[comment] = comment
    logger.info(f"Summarized {len(comments)} comments.")
    return summaries_dict

def compute_keyword_embeddings(categories: Dict[str, Dict[str, List[str]]], model: SentenceTransformer) -> Dict[Tuple[str, str, str], np.ndarray]:
    """Compute embeddings for category keywords."""
    keyword_embeddings = {}
    for category, subcategories in categories.items():
        for subcategory, keywords in subcategories.items():
            for keyword in keywords:
                key = (category, subcategory, keyword)
                if key not in keyword_embeddings:
                    embedding = model.encode(keyword, show_progress_bar=False)
                    keyword_embeddings[key] = embedding
    return keyword_embeddings

def categorize_comments(feedback_data: pd.DataFrame, categories: Dict[str, Dict[str, List[str]]], similarity_threshold: float, 
                       emerging_issue_mode: bool, model: SentenceTransformer, clustering_algo: str = "kmeans", max_clusters: int = 10, 
                       eps: float = 0.5, min_samples: int = 2) -> pd.DataFrame:
    """Categorize comments using similarity or clustering."""
    keyword_embeddings = compute_keyword_embeddings(categories, model)
    keyword_matrix = np.array(list(keyword_embeddings.values()))
    keyword_mapping = list(keyword_embeddings.keys())
    comment_embeddings = model.encode(feedback_data['summarized_comments'].tolist(), show_progress_bar=False)
    similarity_matrix = cosine_similarity(comment_embeddings, keyword_matrix)
    max_scores = similarity_matrix.max(axis=1)
    max_indices = similarity_matrix.argmax(axis=1)
    categories_list, subcats_list, keyphrases_list = [], [], []
    for score, idx in zip(max_scores, max_indices):
        cat, subcat, kw = keyword_mapping[idx]
        if emerging_issue_mode and score < similarity_threshold:
            cat, subcat = 'Emerging Issue', 'Emerging Issue'
        categories_list.append(cat)
        subcats_list.append(subcat)
        keyphrases_list.append(kw)
    feedback_data['Category'] = categories_list
    feedback_data['Sub-Category'] = subcats_list
    feedback_data['Keyphrase'] = keyphrases_list
    feedback_data['Best Match Score'] = max_scores
    
    if emerging_issue_mode:
        emerging_idx = feedback_data['Category'] == 'Emerging Issue'
        if emerging_idx.sum() > 0:
            emerging_embeddings = comment_embeddings[emerging_idx]
            if clustering_algo == "kmeans":
                kmeans = KMeans(n_clusters=min(max_clusters, emerging_idx.sum()), random_state=42)
                clusters = kmeans.fit_predict(normalize(emerging_embeddings))
            else:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = dbscan.fit_predict(normalize(emerging_embeddings))
            feedback_data.loc[emerging_idx, 'Sub-Category'] = [f"Cluster {c}" for c in clusters]
    logger.info(f"Categorized {len(feedback_data)} comments.")
    return feedback_data

# --- Data Processing ---
@st.cache_data(persist="disk", hash_funcs={dict: lambda x: str(sorted(x.items()))})
def process_feedback_data(feedback_data: pd.DataFrame, comment_column: str, date_column: str, categories: Dict[str, Dict[str, List[str]]], 
                         similarity_threshold: float, emerging_issue_mode: bool, summary_model: str, sentiment_model_name: str, 
                         summary_max_length: int, summary_min_length: int, summarize_threshold: int, clustering_algo: str, 
                         max_clusters: int = 10, eps: float = 0.5, min_samples: int = 2, num_beams: int = 4, custom_regex: Optional[str] = None) -> pd.DataFrame:
    """Process feedback data with summarization, categorization, and sentiment analysis."""
    logger.info("Starting feedback data processing.")
    start_time = time.time()
    if comment_column not in feedback_data.columns or date_column not in feedback_data.columns:
        raise ValueError(f"Missing column(s): '{comment_column}' or '{date_column}'.")
    
    model = initialize_bert_model()
    sentiment_model = get_sentiment_model(sentiment_model_name)
    summaries_dict = preprocess_comments_and_summarize(feedback_data, comment_column, summary_model, 
                                                      max_length=summary_max_length, min_length=summary_min_length, 
                                                      summarize_threshold=summarize_threshold, num_beams=num_beams, custom_regex=custom_regex)
    feedback_data['preprocessed_comments'] = feedback_data[comment_column].apply(lambda x: preprocess_text(x, custom_regex))
    feedback_data['summarized_comments'] = feedback_data['preprocessed_comments'].map(summaries_dict)
    feedback_data['Sentiment'] = feedback_data['preprocessed_comments'].apply(lambda x: perform_sentiment_analysis(x, sentiment_model, use_nltk=False))
    feedback_data['Token Count'] = feedback_data['preprocessed_comments'].apply(lambda x: get_token_count(x, get_summarization_model_and_tokenizer(summary_model)[1]))
    feedback_data = categorize_comments(feedback_data, categories, similarity_threshold, emerging_issue_mode, model, clustering_algo, max_clusters, eps, min_samples)
    feedback_data['Parsed Date'] = pd.to_datetime(feedback_data[date_column], errors='coerce')
    feedback_data['Hour'] = feedback_data['Parsed Date'].dt.hour
    feedback_data['Day of Week'] = feedback_data['Parsed Date'].dt.day_name()
    
    logger.info(f"Data processing completed in {time.time() - start_time:.2f} seconds.")
    return feedback_data

# --- Export Functions ---
def export_to_csv(df: pd.DataFrame, filename: str) -> str:
    """Export DataFrame to a CSV file."""
    output = StringIO()
    df.to_csv(output, index=False)
    return save_local_file(output.getvalue().encode(), filename)

def export_to_excel(df: pd.DataFrame, filename: str) -> str:
    """Export DataFrame to an Excel file."""
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Feedback')
    writer.close()
    return save_local_file(output.getvalue(), filename)

def export_to_json(df: pd.DataFrame, filename: str) -> str:
    """Export DataFrame to a JSON file."""
    json_data = df.to_json(orient='records')
    return save_local_file(json_data.encode(), filename)

# --- Streamlit Application ---
def main():
    """Main function for the local Streamlit application."""
    st.set_page_config(layout="wide", page_title="Local Transcript Analysis Dashboard", initial_sidebar_state="expanded")

    # Sidebar Configuration
    st.sidebar.header("Preferences")
    language = st.sidebar.selectbox("Language", list(SUPPORTED_LANGUAGES.keys()), format_func=lambda x: SUPPORTED_LANGUAGES[x])
    theme = st.sidebar.selectbox("Theme", list(THEME_OPTIONS.keys()))
    if theme == "Dark":
        st.markdown("<style>body {background-color: #1E1E1E; color: #FFFFFF;} .stApp {background-color: #1E1E1E; color: #FFFFFF;}</style>", unsafe_allow_html=True)
    elif theme == "Custom":
        primary_color = st.sidebar.color_picker("Primary Color", "#FF4B4B")
        background_color = st.sidebar.color_picker("Background Color", "#FFFFFF")
        st.markdown(f"<style>body {{background-color: {background_color};}} .stApp {{background-color: {background_color};}} .stButton>button {{background-color: {primary_color}; color: white;}}</style>", unsafe_allow_html=True)

    # Title and Navigation
    st.title(f"📊 Local Transcript Analysis Dashboard ({SUPPORTED_LANGUAGES[language]})")
    with st.sidebar:
        menu_selection = option_menu(
            "Menu", ["Home", "Analysis", "Performance"],
            icons=['house', 'bar-chart', 'speedometer'],
            menu_icon="cast", default_index=0,
            styles={"container": {"background-color": "#fafafa"}, "icon": {"color": "orange"}, "nav-link-selected": {"background-color": "#FF4B4B"}}
        )

    # Home Section
    if menu_selection == "Home":
        st.markdown("""
            ### Welcome to the Local Transcript Analysis Dashboard
            This tool runs locally after initial model downloads and includes:
            - **Categorization**: Assign comments to predefined or emerging categories.
            - **Sentiment Analysis**: Evaluate comment sentiment.
            - **Summarization**: Summarize long comments.
            - **Visualizations**: Generate charts and word clouds.
            - **Exports**: Save results as CSV, Excel, or JSON.
            Upload a CSV file in the Analysis section to begin.
        """)

    # Analysis Section
    elif menu_selection == "Analysis":
        st.header("Data Analysis")
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
        
        # Configuration
        st.sidebar.header("⚙️ Analysis Settings")
        similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.35)
        emerging_issue_mode = st.sidebar.checkbox("Emerging Issue Detection", value=True)
        chunk_size = st.sidebar.number_input("Chunk Size", min_value=32, value=32, step=32)
        clustering_algo = st.sidebar.selectbox("Clustering Algorithm", ["kmeans", "dbscan"])
        max_clusters = st.sidebar.number_input("Max Clusters", min_value=1, max_value=50, value=10) if clustering_algo == "kmeans" else 10
        eps = st.sidebar.slider("DBSCAN Epsilon", 0.1, 2.0, 0.5) if clustering_algo == "dbscan" else 0.5
        min_samples = st.sidebar.number_input("DBSCAN Min Samples", min_value=2, value=5) if clustering_algo == "dbscan" else 2
        
        st.sidebar.header("Summarization Settings")
        summary_model = st.sidebar.selectbox("Model", ["knkarthick/MEETING_SUMMARY", "facebook/bart-large-cnn"])
        summary_max_length = st.sidebar.number_input("Max Length", min_value=10, value=75)
        summary_min_length = st.sidebar.number_input("Min Length", min_value=5, value=30)
        summarize_threshold = st.sidebar.number_input("Threshold (tokens)", min_value=10, value=30)
        num_beams = st.sidebar.number_input("Beam Search Width", min_value=1, value=4)
        custom_regex = st.sidebar.text_input("Custom Regex (optional)")
        
        st.sidebar.header("Sentiment Settings")
        sentiment_model_name = st.sidebar.selectbox("Model", ["distilbert-base-uncased-finetuned-sst-2-english", "nlptown/bert-base-multilingual-uncased-sentiment"])

        if uploaded_file:
            csv_data = uploaded_file.read()
            encoding = chardet.detect(csv_data)['encoding']
            uploaded_file.seek(0)
            total_rows = sum(1 for _ in uploaded_file) - 1
            uploaded_file.seek(0)
            total_chunks = math.ceil(total_rows / chunk_size)
            df = pd.read_csv(BytesIO(csv_data), encoding=encoding)
            column_names = df.columns.tolist()

            comment_column = st.selectbox("Comment Column", column_names)
            date_column = st.selectbox("Date Column", column_names)
            grouping_option = st.radio("Group By", ["Date", "Week", "Month", "Quarter", "Hour"])

            if st.button("Process Data"):
                chunk_iter = pd.read_csv(BytesIO(csv_data), encoding=encoding, chunksize=chunk_size)
                processed_chunks = []
                progress_bar = st.progress(0)
                for i, chunk in enumerate(chunk_iter):
                    try:
                        processed_chunk = process_feedback_data(
                            chunk, comment_column, date_column, default_categories,
                            similarity_threshold, emerging_issue_mode, summary_model,
                            sentiment_model_name, summary_max_length, summary_min_length,
                            summarize_threshold, clustering_algo, max_clusters, eps, min_samples, num_beams, custom_regex
                        )
                        processed_chunks.append(processed_chunk)
                        progress_bar.progress((i + 1) / total_chunks)
                    except Exception as e:
                        st.error(f"Error in chunk {i+1}: {e}")
                        logger.error(f"Chunk processing error: {e}")
                
                if processed_chunks:
                    trends_data = pd.concat(processed_chunks, ignore_index=True)
                    st.session_state.trends_data = trends_data
                    
                    # Filters and Visualizations
                    st.subheader("Filter Data")
                    categories_filter = st.multiselect("Categories", options=trends_data['Category'].unique())
                    sentiment_range = st.slider("Sentiment Range", -1.0, 1.0, (-1.0, 1.0))
                    filtered_data = trends_data[
                        (trends_data['Category'].isin(categories_filter) if categories_filter else True) &
                        (trends_data['Sentiment'].between(sentiment_range[0], sentiment_range[1]))
                    ]
                    st.dataframe(filtered_data)

                    # Visualizations
                    viz_type = st.selectbox("Visualization Type", ["Line Chart", "Word Cloud", "Bar Chart"])
                    if viz_type == "Line Chart":
                        pivot_trends = filtered_data.groupby([pd.Grouper(key='Parsed Date', freq=grouping_option[0]), 'Sub-Category']).size().unstack(fill_value=0)
                        top_subcats = pivot_trends.sum().nlargest(5).index
                        fig = px.line(pivot_trends[top_subcats].reset_index(), x='Parsed Date', y=top_subcats, title="Trends Over Time")
                        st.plotly_chart(fig)
                    elif viz_type == "Word Cloud":
                        text = " ".join(filtered_data['preprocessed_comments'])
                        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
                        st.image(wordcloud.to_array())
                    elif viz_type == "Bar Chart":
                        bar_data = filtered_data['Category'].value_counts()
                        fig = px.bar(x=bar_data.index, y=bar_data.values, title="Category Distribution")
                        st.plotly_chart(fig)

                    # Export Options
                    st.subheader("Export Data")
                    export_format = st.selectbox("Format", ["CSV", "Excel", "JSON"])
                    export_filename = st.text_input("Export Filename", value="feedback_export")
                    if st.button("Export"):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        full_filename = f"{export_filename}_{timestamp}"
                        if export_format == "CSV":
                            filepath = export_to_csv(filtered_data, f"{full_filename}.csv")
                            st.download_button("Download CSV", open(filepath, 'rb').read(), f"{full_filename}.csv", "text/csv")
                        elif export_format == "Excel":
                            filepath = export_to_excel(filtered_data, f"{full_filename}.xlsx")
                            st.download_button("Download Excel", open(filepath, 'rb').read(), f"{full_filename}.xlsx", "application/vnd.ms-excel")
                        else:
                            filepath = export_to_json(filtered_data, f"{full_filename}.json")
                            st.download_button("Download JSON", open(filepath, 'rb').read(), f"{full_filename}.json", "application/json")

    # Performance Section
    elif menu_selection == "Performance":
        st.header("Performance Metrics")
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        cpu_usage = process.cpu_percent(interval=1)
        disk_usage = shutil.disk_usage(".").used / 1024 / 1024  # MB
        
        st.metric("Memory Usage (MB)", f"{memory_usage:.2f}")
        st.metric("CPU Usage (%)", f"{cpu_usage:.2f}")
        st.metric("Disk Usage (MB)", f"{disk_usage:.2f}")

if __name__ == "__main__":
    main()
