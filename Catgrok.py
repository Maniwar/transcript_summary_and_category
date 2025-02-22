import os
import time
import math
import base64
import json
import logging
import queue
import threading
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
from sklearn.feature_extraction.text import TfidfVectorizer
import chardet
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st
import textwrap
from categories_josh_sub_V6_3 import default_categories  # Assume this exists locally
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import xlsxwriter
from faker import Faker  # For simulating real-time data
from wordcloud import WordCloud
from gensim import corpora, models
import re
import string
from concurrent.futures import ThreadPoolExecutor

# Set environment variables for local operation
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Download required NLTK data locally (runs once during setup)
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Configure logging to local file
logging.basicConfig(
    filename="transcript_app_local_noauth.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Constants and Configurations ---
SUPPORTED_LANGUAGES = {"en": "English"}  # Limited to English for local operation
THEME_OPTIONS = {"Light": "light", "Dark": "dark", "Custom": "custom"}
LOCAL_STORAGE_DIR = "local_storage_noauth"
STOPWORDS = set(stopwords.words('english'))

# Ensure local directory exists
os.makedirs(LOCAL_STORAGE_DIR, exist_ok=True)

# --- Device Detection ---
def get_device() -> str:
    """
    Determine the best available device for computation locally.

    Returns:
        str: Device type ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        logger.info("CUDA device detected for local use.")
        return 'cuda'
    elif torch.backends.mps.is_available():
        logger.info("MPS device detected for local use.")
        return 'mps'
    else:
        logger.info("CPU device fallback for local use.")
        return 'cpu'

# --- Model Initialization ---
@st.cache_resource
def initialize_bert_model() -> SentenceTransformer:
    """
    Initialize and cache the BERT model for local embedding generation.

    Downloads the model on first run and stores it locally for offline use.

    Returns:
        SentenceTransformer: Initialized BERT model.
    """
    start_time = time.time()
    device = get_device()
    model = SentenceTransformer('all-mpnet-base-v2', device=device)
    logger.info(f"BERT model initialized locally on {device} in {time.time() - start_time:.2f} seconds.")
    return model

@st.cache_resource
def get_summarization_model_and_tokenizer(model_name: str = "knkarthick/MEETING_SUMMARY") -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer, str]:
    """
    Initialize and cache the summarization model and tokenizer locally.

    Downloads the model on first run and stores it locally for offline use.

    Args:
        model_name (str): Name of the summarization model (default: 'knkarthick/MEETING_SUMMARY').

    Returns:
        Tuple[AutoModelForSeq2SeqLM, AutoTokenizer, str]: Model, tokenizer, and device.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = get_device()
    model.to(device)
    logger.info(f"Summarization model {model_name} loaded locally on {device}.")
    return model, tokenizer, device

@st.cache_resource
def get_sentiment_model(model_name: str = "distilbert-base-uncased-finetuned-sst-2-english") -> pipeline:
    """
    Initialize and cache the sentiment analysis model locally.

    Downloads the model on first run and stores it locally for offline use.

    Args:
        model_name (str): Name of the sentiment model (default: 'distilbert-base-uncased-finetuned-sst-2-english').

    Returns:
        pipeline: Sentiment analysis pipeline.
    """
    device = 0 if torch.cuda.is_available() else -1
    model = pipeline("sentiment-analysis", model=model_name, device=device)
    logger.info(f"Sentiment model {model_name} initialized locally.")
    return model

# --- Utility Functions ---
def preprocess_text(text: str, custom_regex: Optional[str] = None) -> str:
    """
    Preprocess text by removing special characters and normalizing whitespace locally.

    Args:
        text (str): Input text to preprocess.
        custom_regex (Optional[str]): Custom regex pattern for additional cleaning (default: None).

    Returns:
        str: Preprocessed text.
    """
    if pd.isna(text):
        return ""
    text = str(text).encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.replace('\n', ' ').replace('\r', '').replace(' ', ' ')
    if custom_regex:
        text = re.sub(custom_regex, '', text)
    processed_text = re.sub(r'\s+', ' ', text).strip()
    logger.debug(f"Preprocessed text locally: {processed_text[:50]}...")
    return processed_text

def save_local_file(content: bytes, filename: str, directory: str = LOCAL_STORAGE_DIR) -> str:
    """
    Save content to a local file.

    Args:
        content (bytes): Content to save.
        filename (str): Name of the file.
        directory (str): Directory to save in (default: LOCAL_STORAGE_DIR).

    Returns:
        str: Path to the saved file.
    """
    filepath = os.path.join(directory, filename)
    try:
        with open(filepath, 'wb') as f:
            f.write(content)
        logger.info(f"Saved file locally: {filepath}")
        return filepath
    except IOError as e:
        logger.error(f"Local file save error: {e}")
        raise IOError(f"Failed to save file {filename}: {e}")

def load_local_file(filepath: str) -> bytes:
    """
    Load content from a local file.

    Args:
        filepath (str): Path to the file.

    Returns:
        bytes: File content.
    """
    try:
        with open(filepath, 'rb') as f:
            content = f.read()
        logger.debug(f"Loaded local file: {filepath}")
        return content
    except IOError as e:
        logger.error(f"Local file load error: {e}")
        raise IOError(f"Failed to load file {filepath}: {e}")

def perform_sentiment_analysis(text: str, sentiment_model=None, use_nltk: bool = True) -> float:
    """
    Perform sentiment analysis locally using NLTK's VADER or a preloaded transformer model.

    Args:
        text (str): Input text for sentiment analysis.
        sentiment_model: Preloaded sentiment model (optional).
        use_nltk (bool): Whether to use NLTK's VADER (default: True).

    Returns:
        float: Sentiment score between -1 and 1.
    """
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
            logger.error(f"Local sentiment analysis error: {e}")
            score = 0.0
    logger.debug(f"Local sentiment score for '{text[:50]}...': {score}")
    return score

def get_token_count(text: str, tokenizer: AutoTokenizer) -> int:
    """
    Count tokens in the text locally using the provided tokenizer.

    Args:
        text (str): Input text.
        tokenizer (AutoTokenizer): Tokenizer instance.

    Returns:
        int: Number of tokens.
    """
    try:
        count = len(tokenizer.encode(text)) - 2
        logger.debug(f"Local token count for '{text[:50]}...': {count}")
        return count
    except Exception as e:
        logger.error(f"Local token count error: {e}")
        return 0

def split_comments_into_chunks(comments: List[Tuple[str, int]], tokenizer: AutoTokenizer, max_tokens: int = 1000) -> List[str]:
    """
    Split comments into chunks based on token limits locally.

    Args:
        comments (List[Tuple[str, int]]): List of (comment, token_count) tuples.
        tokenizer (AutoTokenizer): Tokenizer instance.
        max_tokens (int): Maximum tokens per chunk (default: 1000).

    Returns:
        List[str]: List of chunked comments.
    """
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
    logger.info(f"Locally split {len(sorted_comments)} comments into {len(chunks)} chunks.")
    return chunks

def summarize_text_batch(texts: List[str], tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM, device: str, max_length: int = 75, min_length: int = 30, num_beams: int = 4) -> List[str]:
    """
    Summarize a batch of texts locally using the provided model and tokenizer.

    Args:
        texts (List[str]): List of texts to summarize.
        tokenizer (AutoTokenizer): Tokenizer instance.
        model (AutoModelForSeq2SeqLM): Summarization model.
        device (str): Device to run the model on.
        max_length (int): Maximum summary length (default: 75).
        min_length (int): Minimum summary length (default: 30).
        num_beams (int): Number of beams for beam search (default: 4).

    Returns:
        List[str]: List of summaries.
    """
    try:
        inputs = tokenizer(texts, truncation=True, padding=True, max_length=1024, return_tensors='pt').to(device)
        summary_ids = model.generate(inputs['input_ids'], max_length=max_length, min_length=min_length, num_beams=num_beams)
        summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
        logger.debug(f"Locally summarized {len(texts)} texts.")
        return summaries
    except Exception as e:
        logger.error(f"Local summarization error: {e}")
        return ["Error"] * len(texts)

def preprocess_comments_and_summarize(feedback_data: pd.DataFrame, comment_column: str, model_name: str, batch_size: int = 32, 
                                    max_length: int = 75, min_length: int = 30, max_tokens: int = 1000, summarize_threshold: int = 30, 
                                    num_beams: int = 4, custom_regex: Optional[str] = None) -> Dict[str, str]:
    """
    Preprocess and summarize comments locally with parallel processing and custom options.

    Args:
        feedback_data (pd.DataFrame): DataFrame containing feedback data.
        comment_column (str): Column name for comments.
        model_name (str): Name of the locally cached summarization model.
        batch_size (int): Batch size for local summarization (default: 32).
        max_length (int): Maximum summary length (default: 75).
        min_length (int): Minimum summary length (default: 30).
        max_tokens (int): Maximum tokens per chunk (default: 1000).
        summarize_threshold (int): Token threshold for local summarization (default: 30).
        num_beams (int): Number of beams for local beam search (default: 4).
        custom_regex (Optional[str]): Custom regex for additional local cleaning (default: None).

    Returns:
        Dict[str, str]: Mapping of original comments to summaries.
    """
    if comment_column not in feedback_data.columns:
        raise ValueError(f"Comment column '{comment_column}' not found in local CSV.")
    model, tokenizer, device = get_summarization_model_and_tokenizer(model_name)
    feedback_data['preprocessed_comments'] = feedback_data[comment_column].apply(lambda x: preprocess_text(x, custom_regex))
    comments = feedback_data['preprocessed_comments'].tolist()
    token_counts = [get_token_count(c, tokenizer) for c in comments]
    summaries_dict = {}
    
    logger.info(f"Local comments not summarized (tokens <= {summarize_threshold}): {sum(1 for tc in token_counts if tc <= summarize_threshold)}")
    logger.info(f"Local comments to summarize: {sum(1 for tc in token_counts if tc > summarize_threshold)}")
    
    to_summarize = [(c, tc) for c, tc in zip(comments, token_counts) if tc > summarize_threshold]
    
    def summarize_batch(batch: List[Tuple[str, int]]) -> List[Tuple[str, str]]:
        """Helper function for local parallel summarization."""
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
    logger.info(f"Completed local summarization for {len(comments)} comments.")
    return summaries_dict

def extract_keywords(texts: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Extract top keywords locally using TF-IDF.

    Args:
        texts (List[str]): List of texts to analyze.
        top_n (int): Number of top keywords to return (default: 10).

    Returns:
        List[Tuple[str, float]]: List of (keyword, score) tuples.
    """
    vectorizer = TfidfVectorizer(stop_words=list(STOPWORDS), max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.max(axis=0).toarray()[0]
    keywords = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    logger.info(f"Extracted {len(keywords)} local keywords.")
    return keywords

def perform_topic_modeling(texts: List[str], num_topics: int = 5) -> List[List[Tuple[str, float]]]:
    """
    Perform local topic modeling using LDA.

    Args:
        texts (List[str]): List of texts to analyze.
        num_topics (int): Number of topics to extract (default: 5).

    Returns:
        List[List[Tuple[str, float]]]: List of topics, each with top words and their probabilities.
    """
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]
    filtered_texts = [[word for word in tokens if word not in STOPWORDS and word.isalnum()] for tokens in tokenized_texts]
    dictionary = corpora.Dictionary(filtered_texts)
    corpus = [dictionary.doc2bow(text) for text in filtered_texts]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    topics = [lda_model.show_topic(i, topn=10) for i in range(num_topics)]
    logger.info(f"Performed local topic modeling with {num_topics} topics.")
    return topics

def compute_keyword_embeddings(categories: Dict[str, Dict[str, List[str]]], model: SentenceTransformer) -> Dict[Tuple[str, str, str], np.ndarray]:
    """
    Compute embeddings for category keywords locally.

    Args:
        categories (Dict[str, Dict[str, List[str]]]): Nested dictionary of categories.
        model (SentenceTransformer): Local embedding model.

    Returns:
        Dict[Tuple[str, str, str], np.ndarray]: Mapping of (category, subcategory, keyword) to embeddings.
    """
    keyword_embeddings = {}
    for category, subcategories in categories.items():
        for subcategory, keywords in subcategories.items():
            for keyword in keywords:
                key = (category, subcategory, keyword)
                if key not in keyword_embeddings:
                    embedding = model.encode(keyword, show_progress_bar=False)
                    keyword_embeddings[key] = embedding
                    logger.debug(f"Computed local embedding for keyword: {keyword}")
    logger.info(f"Computed local embeddings for {len(keyword_embeddings)} keywords.")
    return keyword_embeddings

def categorize_comments(feedback_data: pd.DataFrame, categories: Dict[str, Dict[str, List[str]]], similarity_threshold: float, 
                       emerging_issue_mode: bool, model: SentenceTransformer, clustering_algo: str = "kmeans", max_clusters: int = 10, 
                       eps: float = 0.5, min_samples: int = 2) -> pd.DataFrame:
    """
    Categorize comments locally using similarity or clustering.

    Args:
        feedback_data (pd.DataFrame): DataFrame with summarized comments.
        categories (Dict[str, Dict[str, List[str]]]): Nested dictionary of categories.
        similarity_threshold (float): Similarity threshold for local categorization.
        emerging_issue_mode (bool): Whether to detect emerging issues locally.
        model (SentenceTransformer): Local embedding model.
        clustering_algo (str): Clustering algorithm ('kmeans' or 'dbscan', default: 'kmeans').
        max_clusters (int): Maximum number of clusters for KMeans (default: 10).
        eps (float): DBSCAN epsilon parameter (default: 0.5).
        min_samples (int): DBSCAN minimum samples parameter (default: 2).

    Returns:
        pd.DataFrame: DataFrame with categorized comments.
    """
    keyword_embeddings = compute_keyword_embeddings(categories, model)
    keyword_matrix = np.array(list(keyword_embeddings.values()))
    keyword_mapping = list(keyword_embeddings.keys())
    batch_size = 1024
    comment_embeddings = []
    comments = feedback_data['summarized_comments'].tolist()
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i + batch_size]
        embeddings = model.encode(batch, show_progress_bar=False)
        comment_embeddings.extend(embeddings)
    comment_matrix = np.array(comment_embeddings)
    similarity_matrix = cosine_similarity(comment_matrix, keyword_matrix)
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
            emerging_embeddings = comment_matrix[emerging_idx]
            if clustering_algo == "kmeans":
                kmeans = KMeans(n_clusters=min(max_clusters, emerging_idx.sum()), random_state=42)
                clusters = kmeans.fit_predict(normalize(emerging_embeddings))
            elif clustering_algo == "dbscan":
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = dbscan.fit_predict(normalize(emerging_embeddings))
            feedback_data.loc[emerging_idx, 'Sub-Category'] = [f"Cluster {c}" for c in clusters]
    logger.info(f"Locally categorized {len(comments)} comments.")
    return feedback_data

# --- Data Processing ---
@st.cache_data(persist="disk", hash_funcs={dict: lambda x: str(sorted(x.items()))})
def process_feedback_data(feedback_data: pd.DataFrame, comment_column: str, date_column: str, categories: Dict[str, Dict[str, List[str]]], 
                         similarity_threshold: float, emerging_issue_mode: bool, summary_model: str, sentiment_model_name: str, 
                         summary_max_length: int, summary_min_length: int, summarize_threshold: int, clustering_algo: str, 
                         max_clusters: int = 10, eps: float = 0.5, min_samples: int = 2, num_beams: int = 4, custom_regex: Optional[str] = None) -> pd.DataFrame:
    """
    Process feedback data locally with summarization, categorization, sentiment analysis, and additional analytics.

    Args:
        feedback_data (pd.DataFrame): Local DataFrame containing feedback data.
        comment_column (str): Local column name for comments.
        date_column (str): Local column name for dates.
        categories (Dict[str, Dict[str, List[str]]]): Local nested dictionary of categories.
        similarity_threshold (float): Local similarity threshold for categorization.
        emerging_issue_mode (bool): Whether to detect emerging issues locally.
        summary_model (str): Name of the locally cached summarization model.
        sentiment_model_name (str): Name of the locally cached sentiment model.
        summary_max_length (int): Maximum summary length.
        summary_min_length (int): Minimum summary length.
        summarize_threshold (int): Local token threshold for summarization.
        clustering_algo (str): Local clustering algorithm ('kmeans' or 'dbscan').
        max_clusters (int): Maximum number of clusters for KMeans (default: 10).
        eps (float): DBSCAN epsilon parameter (default: 0.5).
        min_samples (int): DBSCAN minimum samples parameter (default: 2).
        num_beams (int): Number of beams for local beam search (default: 4).
        custom_regex (Optional[str]): Custom regex for additional local cleaning (default: None).

    Returns:
        pd.DataFrame: Locally processed DataFrame with additional columns.
    """
    logger.info("Starting local feedback data processing.")
    start_time = time.time()
    if comment_column not in feedback_data.columns or date_column not in feedback_data.columns:
        raise ValueError(f"Missing required local column(s): '{comment_column}' or '{date_column}'.")
    
    # Profile the processing locally
    profiler = cProfile.Profile()
    profiler.enable()
    
    model = initialize_bert_model()
    sentiment_model = get_sentiment_model(sentiment_model_name)
    summaries_dict = preprocess_comments_and_summarize(feedback_data, comment_column, summary_model, 
                                                     max_length=summary_max_length, min_length=summary_min_length, summarize_threshold=summarize_threshold,
                                                     num_beams=num_beams, custom_regex=custom_regex)
    feedback_data['preprocessed_comments'] = feedback_data[comment_column].apply(lambda x: preprocess_text(x, custom_regex))
    feedback_data['summarized_comments'] = feedback_data['preprocessed_comments'].map(summaries_dict)
    feedback_data['Sentiment'] = feedback_data['preprocessed_comments'].apply(lambda x: perform_sentiment_analysis(x, sentiment_model, use_nltk=False))
    feedback_data['Token Count'] = feedback_data['preprocessed_comments'].apply(lambda x: get_token_count(x, get_summarization_model_and_tokenizer(summary_model)[1]))
    feedback_data = categorize_comments(feedback_data, categories, similarity_threshold, emerging_issue_mode, model, clustering_algo, max_clusters, eps, min_samples)
    feedback_data['Parsed Date'] = pd.to_datetime(feedback_data[date_column], errors='coerce')
    feedback_data['Hour'] = feedback_data['Parsed Date'].dt.hour
    feedback_data['Day of Week'] = feedback_data['Parsed Date'].dt.day_name()
    
    # Additional local analytics
    feedback_data['Word Count'] = feedback_data['preprocessed_comments'].apply(lambda x: len(x.split()))
    keywords = extract_keywords(feedback_data['preprocessed_comments'].tolist())
    feedback_data['Top Keyword'] = [next((kw for kw, _ in keywords if kw in comment.lower()), "None") for comment in feedback_data['preprocessed_comments']]
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats_file = os.path.join(LOCAL_STORAGE_DIR, f"profiling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(stats_file, 'w') as f:
        stats.stream = f
        stats.print_stats()
    logger.info(f"Local data processing completed in {time.time() - start_time:.2f} seconds. Profiling saved to {stats_file}")
    return feedback_data

# --- Export Functions ---
def export_to_csv(df: pd.DataFrame, filename: str) -> str:
    """
    Export DataFrame to a local CSV file.

    Args:
        df (pd.DataFrame): Local DataFrame to export.
        filename (str): Name of the local output file.

    Returns:
        str: Path to the saved local CSV file.
    """
    output = StringIO()
    df.to_csv(output, index=False)
    filepath = save_local_file(output.getvalue().encode(), filename)
    return filepath

def export_to_excel(df: pd.DataFrame, filename: str) -> str:
    """
    Export DataFrame to a local Excel file.

    Args:
        df (pd.DataFrame): Local DataFrame to export.
        filename (str): Name of the local output file.

    Returns:
        str: Path to the saved local Excel file.
    """
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Feedback')
    writer.close()
    filepath = save_local_file(output.getvalue(), filename)
    return filepath

def export_to_json(df: pd.DataFrame, filename: str) -> str:
    """
    Export DataFrame to a local JSON file.

    Args:
        df (pd.DataFrame): Local DataFrame to export.
        filename (str): Name of the local output file.

    Returns:
        str: Path to the saved local JSON file.
    """
    json_data = df.to_json(orient='records')
    filepath = save_local_file(json_data.encode(), filename)
    return filepath

def backup_data(df: pd.DataFrame, backup_name: str) -> str:
    """
    Create a local backup of the processed data in JSON format.

    Args:
        df (pd.DataFrame): Local DataFrame to back up.
        backup_name (str): Name of the local backup file.

    Returns:
        str: Path to the saved local backup file.
    """
    filepath = save_local_file(df.to_json().encode(), f"backup_{backup_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    logger.info(f"Created local backup: {filepath}")
    return filepath

# --- Real-time Simulation ---
def simulate_real_time_data(q: queue.Queue, stop_event: threading.Event, rate: float = 1.0, comment_length: int = 200) -> None:
    """
    Simulate real-time feedback data locally at a specified rate with configurable comment length.

    Args:
        q (queue.Queue): Local queue to store simulated data.
        stop_event (threading.Event): Local event to stop simulation.
        rate (float): Data generation rate in seconds (default: 1.0).
        comment_length (int): Maximum characters for simulated comments (default: 200).
    """
    fake = Faker()
    while not stop_event.is_set():
        data = {
            "Comment": fake.text(max_nb_chars=comment_length),
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Rating": np.random.randint(1, 6),
            "Source": fake.word()
        }
        q.put(pd.DataFrame([data]))
        logger.debug(f"Locally simulated data entry: {data['Comment'][:50]}...")
        time.sleep(rate)

# --- Streamlit Application ---
def main():
    """
    Main function for the fully local Streamlit application without user accounts.

    Runs entirely on the local system after initial model downloads, with no internet dependencies thereafter.
    """
    st.set_page_config(layout="wide", page_title="Local Transcript Analysis Dashboard NoAuth", initial_sidebar_state="expanded")

    # Sidebar Configuration
    st.sidebar.header("Local Preferences")
    language = st.sidebar.selectbox("Language", list(SUPPORTED_LANGUAGES.keys()), format_func=lambda x: SUPPORTED_LANGUAGES[x], help="Select the local language (English only).")
    theme = st.sidebar.selectbox("Theme", list(THEME_OPTIONS.keys()), format_func=lambda x: x, help="Select the local theme.")
    if theme == "Dark":
        st.markdown("""
            <style>
            body { background-color: #1E1E1E; color: #FFFFFF; }
            .stApp { background-color: #1E1E1E; color: #FFFFFF; }
            </style>
        """, unsafe_allow_html=True)
    elif theme == "Custom":
        primary_color = st.sidebar.color_picker("Primary Color", "#FF4B4B", help="Choose the local primary color for the UI.")
        background_color = st.sidebar.color_picker("Background Color", "#FFFFFF", help="Choose the local background color for the UI.")
        st.markdown(f"""
            <style>
            body {{ background-color: {background_color}; color: #000000; }}
            .stApp {{ background-color: {background_color}; color: #000000; }}
            .stButton>button {{ background-color: {primary_color}; color: white; }}
            </style>
        """, unsafe_allow_html=True)

    # Title and Navigation
    st.title(f"📊 Local Transcript Categorization and Analysis Dashboard ({SUPPORTED_LANGUAGES[language]})")
    with st.sidebar:
        menu_selection = option_menu(
            "Menu", ["Home", "Analysis", "Real-time", "Performance", "Settings", "Text Editor", "File Explorer", "Help"],
            icons=['house', 'bar-chart', 'clock', 'speedometer', 'gear', 'pencil-square', 'folder', 'question-circle'],
            menu_icon="cast", default_index=0,
            orientation="vertical",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#FF4B4B"},
            }
        )

    # Home Section
    if menu_selection == "Home":
        st.markdown("""
            ### Welcome to the Local Transcript Analysis Dashboard

            This tool runs entirely on your local system after initial setup, requiring no internet connectivity except for downloading models and NLTK data on first run. Features include:

            - **Local Categorization**: Assign comments to predefined or emerging categories using locally cached machine learning models.
            - **Local Sentiment Analysis**: Evaluate comment sentiment using pre-downloaded models.
            - **Local Summarization**: Generate summaries of long comments with configurable thresholds, processed locally.
            - **Real-time Simulation**: Monitor simulated feedback data locally with adjustable rates and comment lengths.
            - **Advanced Local Analysis**: Extract keywords, perform topic modeling, and generate statistical summaries locally.
            - **Enhanced Visualizations**: View line charts, heatmaps, word clouds, and bar charts locally.
            - **Local File Management**: Manage exports and backups through a local file explorer.
            - **Performance Monitoring**: Track local CPU, memory, and disk usage with detailed profiling.
            - **Text Editing**: Edit comments locally with a built-in text editor.
            - **Local Exports**: Save analysis results as CSV, Excel, or JSON files on your system.

            **Setup**: Ensure all models and NLTK data are downloaded on first run. No internet required thereafter.
            **Storage**: All files are saved in the 'local_storage_noauth' directory.
            **Logs**: Processing details are logged locally in 'transcript_app_local_noauth.log'.
        """)

    # Analysis Section
    elif menu_selection == "Analysis":
        st.header("Local Data Analysis")
        uploaded_file = st.sidebar.file_uploader("Upload Local CSV", type="csv", help="Upload a local CSV file with feedback data.")
        
        # Configuration
        st.sidebar.header("⚙️ Local Analysis Settings")
        similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.35, help="Local threshold for categorizing comments.")
        emerging_issue_mode = st.sidebar.checkbox("Emerging Issue Detection", value=True, help="Detect new issues locally.")
        chunk_size = st.sidebar.number_input("Chunk Size", min_value=32, value=32, step=32, help="Number of rows to process per local chunk.")
        clustering_algo = st.sidebar.selectbox("Clustering Algorithm", ["kmeans", "dbscan"], help="Local clustering algorithm for emerging issues.")
        if clustering_algo == "kmeans":
            max_clusters = st.sidebar.number_input("Max Clusters", min_value=1, max_value=50, value=10, help="Maximum number of local KMeans clusters.")
            n_init = st.sidebar.number_input("KMeans N-Init", min_value=1, value=10, step=1, help="Number of local KMeans initializations.")
        else:
            eps = st.sidebar.slider("DBSCAN Epsilon", 0.1, 2.0, 0.5, help="Local DBSCAN clustering epsilon parameter.")
            min_samples = st.sidebar.number_input("DBSCAN Min Samples", min_value=2, value=5, step=1, help="Local DBSCAN minimum samples parameter.")
        
        # Summarization Settings
        st.sidebar.header("Local Summarization")
        summary_model = st.sidebar.selectbox("Model", ["knkarthick/MEETING_SUMMARY", "facebook/bart-large-cnn"], help="Select a locally cached summarization model.")
        summary_max_length = st.sidebar.number_input("Max Length", min_value=10, value=75, step=5, help="Maximum length of local summaries.")
        summary_min_length = st.sidebar.number_input("Min Length", min_value=5, value=30, step=5, help="Minimum length of local summaries.")
        summarize_threshold = st.sidebar.number_input("Threshold (tokens)", min_value=10, value=30, step=5, help="Local token threshold for summarization.")
        num_beams = st.sidebar.number_input("Beam Search Width", min_value=1, value=4, step=1, help="Number of beams for local beam search.")
        custom_regex = st.sidebar.text_input("Custom Regex (optional)", help="Optional regex pattern for additional local text cleaning (e.g., '[0-9]+' to remove numbers).")
        
        # Sentiment Settings
        st.sidebar.header("Local Sentiment Analysis")
        sentiment_model_name = st.sidebar.selectbox("Model", [
            "distilbert-base-uncased-finetuned-sst-2-english",
            "nlptown/bert-base-multilingual-uncased-sentiment"
        ], help="Select a locally cached sentiment analysis model.")
        
        # Category Editing
        st.sidebar.header("📋 Local Categories")
        categories = default_categories.copy()
        new_categories = {}
        for category, subcategories in categories.items():
            with st.sidebar.expander(f"Category: {category}", expanded=False):
                category_name = st.text_input(f"Category Name", value=category, key=f"cat_{category}")
                new_subcategories = {}
                for subcategory, keywords in subcategories.items():
                    subcategory_name = st.text_input(f"Subcategory under {category_name}", value=subcategory, key=f"subcat_{category}_{subcategory}")
                    with st.expander(f"Keywords for {subcategory_name}", expanded=False):
                        keywords_input = st.text_area("Keywords", value="\n".join(keywords), key=f"keywords_{category}_{subcategory}", help="Enter keywords for local categorization.")
                    new_subcategories[subcategory_name] = [kw.strip() for kw in keywords_input.split("\n") if kw.strip()]
                new_categories[category_name] = new_subcategories
        categories = new_categories

        if uploaded_file:
            csv_data = uploaded_file.read()
            encoding = chardet.detect(csv_data)['encoding']
            uploaded_file.seek(0)
            total_rows = sum(1 for _ in uploaded_file) - 1
            uploaded_file.seek(0)
            total_chunks = math.ceil(total_rows / chunk_size)
            df = pd.read_csv(BytesIO(csv_data), encoding=encoding)
            column_names = df.columns.tolist()

            comment_column = st.selectbox("Comment Column", column_names, help="Select the local column containing comments.")
            date_column = st.selectbox("Date Column", column_names, help="Select the local column containing dates.")
            grouping_option = st.radio("Group By", ["Date", "Week", "Month", "Quarter", "Hour"], help="Select grouping for local trends visualization.")

            if st.button("Process Data Locally", help="Start processing the local uploaded data."):
                chunk_iter = pd.read_csv(BytesIO(csv_data), encoding=encoding, chunksize=chunk_size)
                processed_chunks = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                processed_data_placeholder = st.empty()
                
                for i, chunk in enumerate(chunk_iter):
                    try:
                        processed_chunk = process_feedback_data(
                            chunk, comment_column, date_column, categories,
                            similarity_threshold, emerging_issue_mode, summary_model,
                            sentiment_model_name, summary_max_length, summary_min_length,
                            summarize_threshold, clustering_algo,
                            max_clusters if clustering_algo == "kmeans" else 10,
                            eps if clustering_algo == "dbscan" else 0.5,
                            min_samples if clustering_algo == "dbscan" else 2,
                            num_beams, custom_regex
                        )
                        processed_chunks.append(processed_chunk)
                        cumulative_data = pd.concat(processed_chunks, ignore_index=True)
                        processed_data_placeholder.dataframe(cumulative_data)
                        progress_bar.progress((i + 1) / total_chunks)
                        status_text.text(f"Processed local chunk {i+1}/{total_chunks}")
                    except Exception as e:
                        st.error(f"Error in local chunk {i+1}: {e}")
                        logger.error(f"Local chunk processing error: {e}")
                
                if processed_chunks:
                    trends_data = pd.concat(processed_chunks, ignore_index=True)
                    st.session_state.trends_data = trends_data
                    
                    # Backup locally
                    backup_path = backup_data(trends_data, "analysis")
                    st.write(f"Local backup created at: {backup_path}")

                    # Filters
                    st.subheader("Filter Local Data")
                    categories_filter = st.multiselect("Categories", options=trends_data['Category'].unique(), help="Filter by local categories.")
                    subcategories_filter = st.multiselect("Sub-Categories", options=trends_data['Sub-Category'].unique(), help="Filter by local sub-categories.")
                    sentiment_range = st.slider("Sentiment Range", -1.0, 1.0, (-1.0, 1.0), help="Filter by local sentiment score.")
                    date_range = st.date_input("Date Range", [trends_data['Parsed Date'].min(), trends_data['Parsed Date'].max()], help="Filter by local date range.")
                    token_range = st.slider("Token Count Range", int(trends_data['Token Count'].min()), int(trends_data['Token Count'].max()), 
                                          (int(trends_data['Token Count'].min()), int(trends_data['Token Count'].max())), help="Filter by local token count.")
                    
                    filtered_data = trends_data[
                        (trends_data['Category'].isin(categories_filter) if categories_filter else True) &
                        (trends_data['Sub-Category'].isin(subcategories_filter) if subcategories_filter else True) &
                        (trends_data['Sentiment'].between(sentiment_range[0], sentiment_range[1])) &
                        (trends_data['Parsed Date'].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))) &
                        (trends_data['Token Count'].between(token_range[0], token_range[1]))
                    ]
                    
                    st.dataframe(filtered_data, height=400)
                    
                    # Statistical Summaries
                    st.subheader("Local Statistical Summaries")
                    st.write("Sentiment Distribution")
                    st.bar_chart(pd.cut(filtered_data['Sentiment'], bins=[-1, -0.5, 0, 0.5, 1], labels=['Negative', 'Slightly Negative', 'Neutral', 'Positive']).value_counts())
                    st.write(f"Average Sentiment: {filtered_data['Sentiment'].mean():.2f}")
                    st.write(f"Average Word Count: {filtered_data['Word Count'].mean():.2f}")
                    st.write(f"Average Token Count: {filtered_data['Token Count'].mean():.2f}")
                    
                    # Topic Modeling
                    st.subheader("Local Topic Modeling")
                    num_topics = st.slider("Number of Topics", 2, 10, 5, help="Number of topics to extract locally.")
                    if st.button("Run Local Topic Modeling"):
                        topics = perform_topic_modeling(filtered_data['preprocessed_comments'].tolist(), num_topics)
                        for i, topic in enumerate(topics):
                            st.write(f"Topic {i+1}: {', '.join(f'{word} ({prob:.2f})' for word, prob in topic)}")

                    # Visualization Options
                    viz_type = st.selectbox("Local Visualization Type", ["Line Chart", "Heatmap", "Word Cloud", "Bar Chart", "Box Plot", "Pie Chart"], 
                                          help="Select the type of local visualization.")
                    
                    if viz_type == "Line Chart":
                        if grouping_option != 'Hour':
                            pivot_trends = filtered_data.groupby([pd.Grouper(key='Parsed Date', freq=grouping_option[0]), 'Sub-Category']).size().unstack(fill_value=0)
                            top_subcats = pivot_trends.sum().nlargest(5).index
                            fig = px.line(pivot_trends[top_subcats].reset_index(), x='Parsed Date', y=top_subcats, title="Local Trends Over Time")
                            st.plotly_chart(fig)
                        else:
                            pivot_trends = filtered_data.groupby(['Hour', 'Sub-Category']).size().unstack(fill_value=0)
                            top_subcats = pivot_trends.sum().nlargest(5).index
                            fig = px.line(pivot_trends[top_subcats].reset_index(), x='Hour', y=top_subcats, title="Local Trends by Hour")
                            st.plotly_chart(fig)
                    elif viz_type == "Heatmap":
                        heatmap_data = filtered_data.groupby(['Category', 'Sub-Category']).size().unstack(fill_value=0)
                        fig = go.Figure(data=go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index, colorscale='Viridis'))
                        st.plotly_chart(fig)
                    elif viz_type == "Word Cloud":
                        text = " ".join(filtered_data['preprocessed_comments'])
                        background_color = st.color_picker("Word Cloud Background", "#FFFFFF", key="wc_bg")
                        max_words = st.slider("Max Words", 50, 500, 100, help="Maximum words in local word cloud.")
                        wordcloud = WordCloud(width=800, height=400, background_color=background_color, max_words=max_words).generate(text)
                        st.image(wordcloud.to_array())
                    elif viz_type == "Bar Chart":
                        bar_data = filtered_data['Category'].value_counts()
                        fig = px.bar(x=bar_data.index, y=bar_data.values, title="Local Category Distribution")
                        st.plotly_chart(fig)
                    elif viz_type == "Box Plot":
                        fig = px.box(filtered_data, x='Category', y='Sentiment', title="Local Sentiment Distribution by Category")
                        st.plotly_chart(fig)
                    elif viz_type == "Pie Chart":
                        pie_data = filtered_data['Category'].value_counts()
                        fig = px.pie(names=pie_data.index, values=pie_data.values, title="Local Category Proportions")
                        st.plotly_chart(fig)

                    # Local Export Options
                    st.subheader("Export Local Data")
                    export_format = st.selectbox("Format", ["CSV", "Excel", "JSON"], help="Select format for local export.")
                    export_filename = st.text_input("Export Filename", value="feedback_export", help="Enter the base filename for local export.")
                    if st.button("Export Locally", help="Export the processed data to your local system."):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        full_filename = f"{export_filename}_{timestamp}"
                        if export_format == "CSV":
                            filepath = export_to_csv(filtered_data, f"{full_filename}.csv")
                            st.download_button("Download CSV", open(filepath, 'rb').read(), f"{full_filename}.csv", "text/csv")
                            st.write(f"Saved locally at: {filepath}")
                        elif export_format == "Excel":
                            filepath = export_to_excel(filtered_data, f"{full_filename}.xlsx")
                            st.download_button("Download Excel", open(filepath, 'rb').read(), f"{full_filename}.xlsx", "application/vnd.ms-excel")
                            st.write(f"Saved locally at: {filepath}")
                        else:
                            filepath = export_to_json(filtered_data, f"{full_filename}.json")
                            st.download_button("Download JSON", open(filepath, 'rb').read(), f"{full_filename}.json", "application/json")
                            st.write(f"Saved locally at: {filepath}")

    # Real-time Section
    elif menu_selection == "Real-time":
        st.header("Local Real-time Feedback Simulator")
        if "real_time_data" not in st.session_state:
            st.session_state.real_time_data = pd.DataFrame()
        q = queue.Queue()
        stop_event = threading.Event()
        rate = st.sidebar.slider("Data Rate (seconds)", 0.1, 10.0, 1.0, help="Local rate of simulated data generation.")
        comment_length = st.sidebar.slider("Comment Length (chars)", 50, 500, 200, help="Maximum length of local simulated comments.")
        simulation_state = st.sidebar.selectbox("Simulation State", ["Running", "Paused", "Stopped"], help="Control the local real-time simulation.")
        
        if simulation_state == "Running":
            thread = threading.Thread(target=simulate_real_time_data, args=(q, stop_event, rate, comment_length))
            thread.start()
        
        placeholder = st.empty()
        stop_button = st.button("Stop Local Real-time", help="Stop the local real-time simulation.")
        while not stop_button and simulation_state == "Running":
            try:
                new_data = q.get_nowait()
                processed_data = process_feedback_data(
                    new_data, "Comment", "Date", categories, similarity_threshold, emerging_issue_mode,
                    summary_model, sentiment_model_name, summary_max_length, summary_min_length, summarize_threshold, clustering_algo, max_clusters
                )
                st.session_state.real_time_data = pd.concat([st.session_state.real_time_data, processed_data], ignore_index=True)
                placeholder.dataframe(st.session_state.real_time_data)
            except queue.Empty:
                time.sleep(0.1)
        if stop_button or simulation_state == "Stopped":
            stop_event.set()
            thread.join()
            st.write("Local real-time simulation stopped.")

    # Performance Section
    elif menu_selection == "Performance":
        st.header("Local Performance Metrics")
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        cpu_usage = process.cpu_percent(interval=1)
        disk_usage = shutil.disk_usage(".").used / 1024 / 1024  # MB
        
        st.metric("Memory Usage (MB)", f"{memory_usage:.2f}", help="Current local memory usage of the application.")
        st.metric("CPU Usage (%)", f"{cpu_usage:.2f}", help="Current local CPU usage of the application.")
        st.metric("Disk Usage (MB)", f"{disk_usage:.2f}", help="Current local disk usage in the working directory.")
        
        st.subheader("Local Log Viewer")
        log_filter = st.text_input("Filter Log", help="Enter a keyword to filter local log entries.")
        with open("transcript_app_local_noauth.log", "r") as f:
            log_content = f.readlines()
            filtered_log = [line for line in log_content if not log_filter or log_filter.lower() in line.lower()]
            st.text_area("Local Log", "".join(filtered_log), height=300, help="Local application log with filtering.")

    # Settings Section
    elif menu_selection == "Settings":
        st.header("Local Application Settings")
        st.subheader("Local Processing Options")
        st.slider("Thread Pool Workers", 1, 8, 4, key="thread_workers", help="Number of local workers for parallel processing.")
        
        st.subheader("Local Storage Management")
        if st.button("Clear Local Storage", help="Delete all locally stored exports and backups."):
            for file in glob.glob(os.path.join(LOCAL_STORAGE_DIR, "*")):
                os.remove(file)
            st.success("Local storage cleared.")
            logger.info("Local storage cleared by user.")

    # Text Editor Section
    elif menu_selection == "Text Editor":
        st.header("Local Text Editor")
        if "trends_data" in st.session_state and not st.session_state.trends_data.empty:
            st.subheader("Edit Local Comments")
            comment_index = st.number_input("Comment Index", 0, len(st.session_state.trends_data) - 1, 0, help="Select a local comment to edit.")
            current_comment = st.session_state.trends_data['preprocessed_comments'].iloc[comment_index]
            edited_comment = st.text_area("Edit Comment", value=current_comment, height=200, help="Edit the selected local comment.")
            if st.button("Save Edited Comment", help="Save the edited comment locally."):
                st.session_state.trends_data.at[comment_index, 'preprocessed_comments'] = edited_comment
                st.success("Local comment updated.")
                logger.info(f"Local comment at index {comment_index} updated.")
        else:
            st.write("No local data loaded yet. Please process data in the Analysis section first.")

    # File Explorer Section
    elif menu_selection == "File Explorer":
        st.header("Local File Explorer")
        st.subheader("Browse Local Storage")
        files = glob.glob(os.path.join(LOCAL_STORAGE_DIR, "*"))
        selected_file = st.selectbox("Select Local File", files, help="Browse files stored locally.")
        if selected_file:
            if selected_file.endswith('.csv'):
                st.dataframe(pd.read_csv(selected_file))
            elif selected_file.endswith('.xlsx'):
                st.dataframe(pd.read_excel(selected_file))
            elif selected_file.endswith('.json'):
                st.json(json.loads(open(selected_file, 'r').read()))
            if st.button("Delete Local File", help="Delete the selected local file."):
                os.remove(selected_file)
                st.success(f"Deleted {selected_file} locally.")
                logger.info(f"Deleted local file: {selected_file}")

    # Help Section
    elif menu_selection == "Help":
        st.header("Local Help & Documentation")
        st.markdown("""
            ### How to Use the Local Transcript Analysis Dashboard

            This application runs entirely on your local system after initial setup, requiring no internet connectivity except for downloading models and NLTK data on first run. Here’s how to use it:

            1. **Setup**: Ensure all models (BERT, summarization, sentiment) and NLTK data (vader_lexicon, punkt, stopwords) are downloaded locally on first run. No further internet is needed.
            2. **Upload Data**: In the Analysis section, upload a local CSV file with feedback data (ensure it has comment and date columns).
            3. **Configure Settings**: Adjust local settings like similarity thresholds, clustering algorithms, and summarization options in the sidebar.
            4. **Process Data**: Click "Process Data Locally" to analyze the feedback using local models.
            5. **Explore Results**: Use local filters and visualizations (line charts, heatmaps, word clouds, bar charts, box plots, pie charts) to explore categorized and summarized feedback.
            6. **Advanced Analysis**: Extract keywords, perform topic modeling, and view statistical summaries locally.
            7. **Real-time Simulation**: Go to the Real-time section to monitor simulated feedback data locally with adjustable rates and comment lengths.
            8. **Performance**: Check local memory, CPU, and disk usage with detailed profiling in the Performance section.
            9. **Settings**: Customize local processing options and manage storage in the Settings section.
            10. **Text Editor**: Edit comments locally in the Text Editor section.
            11. **File Explorer**: Browse and manage local exports and backups in the File Explorer section.

            **Notes**:
            - All data processing, storage, and visualization occur locally on your system.
            - Ensure sufficient disk space for exports and backups in the 'local_storage_noauth' directory.
            - Logs are saved locally in 'transcript_app_local_noauth.log' for troubleshooting.
            - Example CSV format: Columns 'Comment' (text) and 'Date' (YYYY-MM-DD HH:MM:SS).

            **Example Usage**:
            - Upload a CSV with feedback data.
            - Set summarization threshold to 50, max length to 100, and min length to 20.
            - Enable emerging issue detection with DBSCAN (eps=0.5, min_samples=3).
            - Process the data and explore results with a local heatmap or word cloud.
            - Export to Excel for further local analysis.
        """)

# --- Unit Tests ---
class TestLocalTranscriptAppNoAuth(unittest.TestCase):
    """Unit tests for the local transcript application without authentication."""
    def setUp(self):
        """Set up local test environment."""
        self.test_dir = "test_local_storage"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        """Clean up local test environment."""
        shutil.rmtree(self.test_dir)

    def test_preprocess_text(self):
        """Test local text preprocessing function."""
        text = "Hello, World! \n This is a test..."
        expected = "Hello World This is a test"
        self.assertEqual(preprocess_text(text), expected)

    def test_save_local_file(self):
        """Test saving a local file."""
        content = b"Test content"
        filepath = save_local_file(content, "test.txt", self.test_dir)
        self.assertTrue(os.path.exists(filepath))
        with open(filepath, 'rb') as f:
            self.assertEqual(f.read(), content)

    def test_extract_keywords(self):
        """Test local keyword extraction."""
        texts = ["This is a test", "Test this now", "Another test case"]
        keywords = extract_keywords(texts, top_n=3)
        self.assertEqual(len(keywords), 3)
        self.assertTrue(all(kw[0] in ["test", "case", "this", "now", "another"] for kw in keywords))

if __name__ == "__main__":
    main()
    # Run local tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
