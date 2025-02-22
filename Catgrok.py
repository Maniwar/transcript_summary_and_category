import os
import time
import math
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import chardet
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st
import textwrap
import re
import base64
from io import BytesIO
import xlsxwriter
from tqdm import tqdm
from categories_josh_sub_V6_3 import default_categories

# Set environment variable for tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# --- Model Initialization ---
@st.cache_resource
def initialize_bert_model():
    """Initialize and cache the BERT model."""
    start_time = time.time()
    print("Initializing BERT model...")
    model = SentenceTransformer('all-mpnet-base-v2', device="cpu")
    print(f"BERT model initialized. Time taken: {time.time() - start_time:.2f} seconds.")
    return model

@st.cache_resource
def get_summarization_model_and_tokenizer():
    """Initialize and cache the summarization model and tokenizer."""
    model_name = "knkarthick/MEETING_SUMMARY"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"Summarization model loaded on {device}.")
    return model, tokenizer, device

# --- Utility Functions ---
def preprocess_text(text):
    """Preprocess text by removing special characters and normalizing whitespace."""
    if pd.isna(text) or isinstance(text, float):
        text = str(text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.replace('\n', ' ').replace('\r', '').replace(' ', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def perform_sentiment_analysis(text):
    """Perform sentiment analysis using NLTK's VADER."""
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    return sentiment_scores['compound']

def get_token_count(text, tokenizer):
    """Count tokens in the text using the provided tokenizer."""
    try:
        return len(tokenizer.encode(text)) - 2
    except Exception:
        return 0

def split_comments_into_chunks(comments, tokenizer, max_tokens):
    """Split comments into chunks based on token limits."""
    sorted_comments = sorted(comments, key=lambda x: x[1], reverse=True)
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0
    for comment, tokens in sorted_comments:
        if tokens > max_tokens:
            parts = textwrap.wrap(comment, width=max_tokens // 2)
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
    print(f"Total number of chunks created: {len(chunks)}")
    return chunks

def summarize_text(text, tokenizer, model, device, max_length=75, min_length=30):
    """Summarize a single text using the provided model."""
    input_ids = tokenizer([text], truncation=True, padding=True, return_tensors='pt')['input_ids'].to(device)
    summary_ids = model.generate(input_ids, max_length=max_length, min_length=min_length)[0]
    return tokenizer.decode(summary_ids, skip_special_tokens=True)

def preprocess_comments_and_summarize(feedback_data, comment_column, batch_size=32, max_length=75, min_length=30, max_tokens=1000, very_short_limit=30):
    """Preprocess and summarize comments."""
    print("Starting preprocessing and summarization...")
    feedback_data['preprocessed_comments'] = feedback_data[comment_column].apply(preprocess_text)
    model, tokenizer, device = get_summarization_model_and_tokenizer()
    
    comments = feedback_data['preprocessed_comments'].tolist()
    token_counts = [get_token_count(c, tokenizer) for c in comments]
    very_short_comments = [c for c, tc in zip(comments, token_counts) if tc <= very_short_limit]
    short_comments = [c for c, tc in zip(comments, token_counts) if very_short_limit < tc <= max_tokens]
    long_comments = [c for c, tc in zip(comments, token_counts) if tc > max_tokens]
    
    summaries_dict = {c: c for c in very_short_comments}
    print(f"Separated comments: {len(very_short_comments)} very short, {len(short_comments)} short, {len(long_comments)} long.")
    
    # Summarize short comments in batches
    for i in range(0, len(short_comments), batch_size):
        batch = short_comments[i:i + batch_size]
        summaries = [summarize_text(c, tokenizer, model, device, max_length, min_length) for c in batch]
        summaries_dict.update(zip(batch, summaries))
    
    # Summarize long comments with chunking
    for comment in long_comments:
        chunks = split_comments_into_chunks([(comment, get_token_count(comment, tokenizer))], tokenizer, max_tokens)
        summaries = [summarize_text(chunk, tokenizer, model, device, max_length, min_length) for chunk in chunks]
        full_summary = " ".join(summaries)
        while get_token_count(full_summary, tokenizer) > max_length:
            full_summary = summarize_text(full_summary, tokenizer, model, device, max_length, min_length)
        summaries_dict[comment] = full_summary
    
    print("Preprocessing and summarization completed.")
    return summaries_dict

@st.cache_data(persist="disk")
def compute_keyword_embeddings(categories):
    """Compute embeddings for category keywords."""
    print("Computing keyword embeddings...")
    model = initialize_bert_model()
    keyword_embeddings = {}
    for category, subcategories in categories.items():
        for subcategory, keywords in subcategories.items():
            for keyword in keywords:
                key = (category, subcategory, keyword)
                if key not in keyword_embeddings:
                    keyword_embeddings[key] = model.encode([keyword])[0]
    print(f"Keyword embeddings computed. Time taken: {time.time() - time.time():.2f} seconds.")
    return keyword_embeddings

# --- Data Processing ---
def process_feedback_data(feedback_data, comment_column, date_column, categories, similarity_threshold, emerging_issue_mode):
    """Process feedback data with summarization, categorization, sentiment analysis, and clustering."""
    print("Starting feedback data processing...")
    model = initialize_bert_model()
    
    # Preprocess and summarize comments
    summaries_dict = preprocess_comments_and_summarize(feedback_data, comment_column)
    feedback_data['preprocessed_comments'] = feedback_data[comment_column].apply(preprocess_text)
    feedback_data['summarized_comments'] = feedback_data['preprocessed_comments'].map(summaries_dict).fillna(feedback_data['preprocessed_comments'])
    
    # Compute sentiment scores
    feedback_data['sentiment_scores'] = feedback_data['preprocessed_comments'].apply(perform_sentiment_analysis)
    
    # Compute comment embeddings in batches
    batch_size = 1024
    comment_embeddings = []
    for i in range(0, len(feedback_data), batch_size):
        batch = feedback_data['summarized_comments'][i:i + batch_size].tolist()
        comment_embeddings.extend(model.encode(batch, show_progress_bar=False))
    
    # Categorize comments
    keyword_embeddings = compute_keyword_embeddings(categories)
    keyword_matrix = np.array(list(keyword_embeddings.values()))
    keyword_mapping = list(keyword_embeddings.keys())
    similarity_matrix = cosine_similarity(comment_embeddings, keyword_matrix)
    max_scores = similarity_matrix.max(axis=1)
    max_indices = similarity_matrix.argmax(axis=1)
    
    categories_list = []
    sub_categories_list = []
    keyphrases_list = []
    for score, idx in zip(max_scores, max_indices):
        cat, subcat, kw = keyword_mapping[idx]
        if emerging_issue_mode and score < similarity_threshold:
            cat, subcat = 'No Match', 'No Match'
        categories_list.append(cat)
        sub_categories_list.append(subcat)
        keyphrases_list.append(kw)
    
    # Cluster 'No Match' comments if in emerging issue mode
    if emerging_issue_mode:
        no_match_indices = [i for i, cat in enumerate(categories_list) if cat == 'No Match']
        if no_match_indices:
            no_match_embeddings = np.array([comment_embeddings[i] for i in no_match_indices])
            num_clusters = min(10, len(no_match_indices))
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(no_match_embeddings)
            
            # Generate cluster labels based on centroid proximity
            model_summ, tokenizer_summ, device = get_summarization_model_and_tokenizer()
            cluster_labels = {}
            for cluster_id in range(num_clusters):
                cluster_indices = [no_match_indices[i] for i, c in enumerate(clusters) if c == cluster_id]
                cluster_embeddings = np.array([comment_embeddings[i] for i in cluster_indices])
                centroid = kmeans.cluster_centers_[cluster_id]
                distances = cosine_similarity([centroid], cluster_embeddings)[0]
                closest_idx = cluster_indices[np.argmax(distances)]
                centroid_comment = feedback_data.iloc[closest_idx]['summarized_comments']
                cluster_labels[cluster_id] = summarize_text(centroid_comment, tokenizer_summ, model_summ, device)
            
            # Assign cluster labels
            for idx, cluster in zip(no_match_indices, clusters):
                sub_categories_list[idx] = f"Emerging Issue: {cluster_labels[cluster]}"
    
    # Prepare final DataFrame
    feedback_data['Category'] = categories_list
    feedback_data['Sub-Category'] = sub_categories_list
    feedback_data['Keyphrase'] = keyphrases_list
    feedback_data['Sentiment'] = feedback_data['sentiment_scores']
    feedback_data['Best Match Score'] = max_scores
    feedback_data['Parsed Date'] = pd.to_datetime(feedback_data[date_column], errors='coerce')
    feedback_data['Hour'] = feedback_data['Parsed Date'].dt.hour
    feedback_data.drop(columns=['sentiment_scores'], inplace=True)
    
    print("Feedback data processing completed.")
    return feedback_data

# --- Streamlit Application ---
st.set_page_config(layout="wide")
st.title("👨‍💻 Transcript Categorization")

model = initialize_bert_model()
emerging_issue_mode = st.sidebar.checkbox("Emerging Issue Mode")
st.sidebar.write("Emerging issue mode sets unmatched comments to 'No Match' and clusters them.")
similarity_threshold = st.sidebar.slider("Semantic Similarity Threshold", 0.0, 1.0, 0.35) if emerging_issue_mode else None

# Category Editing
st.sidebar.header("Edit Categories")
new_categories = {}
for category, subcategories in default_categories.items():
    category_name = st.sidebar.text_input(f"{category} Category", value=category)
    new_subcategories = {}
    for subcategory, keywords in subcategories.items():
        subcategory_name = st.sidebar.text_input(f"{subcategory} Subcategory under {category_name}", value=subcategory)
        with st.sidebar.expander(f"Keywords for {subcategory_name}"):
            category_keywords = st.text_area("Keywords", value="\n".join(keywords))
        new_subcategories[subcategory_name] = category_keywords.split("\n")
    new_categories[category_name] = new_subcategories
default_categories = new_categories

# File Upload and Processing
uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file:
    csv_data = uploaded_file.read()
    encoding = chardet.detect(csv_data)['encoding']
    uploaded_file.seek(0)
    chunk_iter = pd.read_csv(BytesIO(csv_data), encoding=encoding, chunksize=32)
    total_rows = sum(1 for _ in pd.read_csv(BytesIO(csv_data), encoding=encoding, chunksize=32)) * 32
    estimated_chunks = math.ceil(total_rows / 32)
    
    column_names = next(chunk_iter).columns.tolist()
    comment_column = st.selectbox("Select the column containing the comments", column_names)
    date_column = st.selectbox("Select the column containing the dates", column_names)
    grouping_option = st.radio("Select how to group the dates", ["Date", "Week", "Month", "Quarter", "Hour"])
    
    if st.button("Process Feedback"):
        progress_bar = st.progress(0)
        processed_chunks = []
        for i, feedback_data in enumerate(pd.read_csv(BytesIO(csv_data), encoding=encoding, chunksize=32)):
            trends_data = process_feedback_data(feedback_data, comment_column, date_column, default_categories, similarity_threshold, emerging_issue_mode)
            processed_chunks.append(trends_data)
            progress_bar.progress((i + 1) / estimated_chunks)
        
        trends_data = pd.concat(processed_chunks, ignore_index=True)
        
        # Visualizations
        st.subheader("Feedback Trends and Insights")
        st.dataframe(trends_data)
        
        pivot = trends_data.pivot_table(
            index=['Category', 'Sub-Category'],
            columns=pd.Grouper(key='Parsed Date', freq='D' if grouping_option == 'Date' else grouping_option[0]),
            values='Sentiment',
            aggfunc='count',
            fill_value=0
        )
        st.subheader("All Categories Trends")
        st.dataframe(pivot)
        st.line_chart(pivot.head(5).T)
        
        pivot1 = trends_data.groupby('Category')['Sentiment'].agg(['mean', 'count']).sort_values('count', ascending=False)
        pivot1.columns = ['Average Sentiment', 'Quantity']
        st.subheader("Category vs Sentiment and Quantity")
        st.dataframe(pivot1)
        st.bar_chart(pivot1['Quantity'])
        
        pivot2 = trends_data.groupby(['Category', 'Sub-Category'])['Sentiment'].agg(['mean', 'count']).sort_values('count', ascending=False)
        pivot2.columns = ['Average Sentiment', 'Quantity']
        st.subheader("Sub-Category vs Sentiment and Quantity")
        st.dataframe(pivot2)
        st.bar_chart(pivot2['Quantity'])
        
        # Top Comments
        top_subcategories = pivot2.index.get_level_values('Sub-Category')[:10]
        st.subheader("Top 10 Most Recent Comments for Each Top Subcategory")
        for subcategory in top_subcategories:
            st.write(f"**{subcategory}**")
            top_comments = trends_data[trends_data['Sub-Category'] == subcategory].nlargest(10, 'Parsed Date')[['Parsed Date', comment_column, 'Summarized Comments', 'Sentiment']]
            st.table(top_comments)
        
        # Excel Export
        excel_file = BytesIO()
        with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
            trends_data.to_excel(writer, sheet_name='Feedback Trends', index=False)
            pivot.to_excel(writer, sheet_name=f'Trends by {grouping_option}', merge_cells=False)
            pivot1.to_excel(writer, sheet_name='Categories', merge_cells=False)
            pivot2.to_excel(writer, sheet_name='Subcategories', merge_cells=False)
        
        excel_file.seek(0)
        b64 = base64.b64encode(excel_file.read()).decode()
        st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="feedback_trends.xlsx">Download Excel File</a>', unsafe_allow_html=True)
