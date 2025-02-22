import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # or "false" to enable/disable parallelism

import torch
from torch.utils.data import Dataset
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import datetime
import numpy as np
import xlsxwriter
import chardet
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import base64
from io import BytesIO
import streamlit as st
import textwrap
import re
import math
from tqdm import tqdm
from collections import defaultdict
import time

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)

# -------------------------------------------
# Optional Dataset class (if needed)
class SummarizationDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(text, truncation=True, padding='max_length',
                                max_length=self.max_length, return_tensors='pt')
        return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze()

# -------------------------------------------
# Caching model initializations
@st.cache_resource
def initialize_bert_model():
    start = time.time()
    st.info("Initializing BERT model...")
    model = SentenceTransformer('all-mpnet-base-v2', device="cpu")
    st.info(f"BERT model initialized in {time.time() - start:.2f} seconds.")
    return model

@st.cache_resource
def get_summarization_model_and_tokenizer():
    model_name = "knkarthick/MEETING_SUMMARY"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return model, tokenizer, device

# Global variable for caching category state
previous_categories = None

@st.cache_data(persist="disk")
def compute_keyword_embeddings(categories):
    start = time.time()
    st.info("Computing keyword embeddings...")
    keyword_embeddings = {}
    for category, subcategories in categories.items():
        for subcategory, keywords in subcategories.items():
            for keyword in keywords:
                key = (category, subcategory, keyword)
                if key not in keyword_embeddings:
                    keyword_embeddings[key] = model.encode([keyword])[0]
    st.info(f"Keyword embeddings computed in {time.time() - start:.2f} seconds.")
    return keyword_embeddings

# -------------------------------------------
# Text preprocessing
def preprocess_text(text):
    if isinstance(text, float):
        text = str(text)
    # 'encoding' variable is defined after file upload
    text = text.encode('ascii', 'ignore').decode(encoding)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.replace('\n', ' ').replace('\r', '')
    text = text.replace('&nbsp;', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------------------------
# Sentiment analysis
def perform_sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)['compound']

# -------------------------------------------
# Token count helper
def get_token_count(text, tokenizer):
    return len(tokenizer.encode(text)) - 2

# -------------------------------------------
# Split long comments into chunks based on token limits
def split_comments_into_chunks(comments, tokenizer, max_tokens):
    # comments: list of tuples (comment, token_count)
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
    print(f"Total chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} token count: {get_token_count(chunk, tokenizer)}")
    return chunks

# -------------------------------------------
# Summarize text function
def summarize_text(text, tokenizer, model, device, max_length=75, min_length=30):
    input_ids = tokenizer([text], truncation=True, padding=True,
                          return_tensors='pt')['input_ids'].to(device)
    summary_ids = model.generate(input_ids, max_length=max_length, min_length=min_length)[0]
    return tokenizer.decode(summary_ids, skip_special_tokens=True)

# -------------------------------------------
# Preprocess and summarize comments for a DataFrame chunk
def preprocess_comments_and_summarize(feedback_data, comment_column, batch_size=32,
                                      max_length=75, min_length=30, max_tokens=1000, very_short_limit=30):
    st.info("Preprocessing and summarizing comments...")
    feedback_data['preprocessed_comments'] = feedback_data[comment_column].apply(preprocess_text)
    st.info("Comments preprocessed.")
    model_summ, tokenizer_summ, device = get_summarization_model_and_tokenizer()
    comments = feedback_data['preprocessed_comments'].tolist()
    very_short_comments = [c for c in comments if get_token_count(c, tokenizer_summ) <= very_short_limit]
    short_comments = [c for c in comments if very_short_limit < get_token_count(c, tokenizer_summ) <= max_tokens]
    long_comments = [c for c in comments if get_token_count(c, tokenizer_summ) > max_tokens]
    st.info(f"Separated comments: {len(very_short_comments)} very short, {len(short_comments)} short, {len(long_comments)} long.")
    summaries_dict = {c: c for c in very_short_comments}
    for i in tqdm(range(0, len(short_comments), batch_size), desc="Summarizing short comments"):
        batch = short_comments[i:i+batch_size]
        summaries = [summarize_text(c, tokenizer_summ, model_summ, device, max_length, min_length) for c in batch]
        summaries_dict.update(dict(zip(batch, summaries)))
    for comment in tqdm(long_comments, desc="Summarizing long comments"):
        chunks = split_comments_into_chunks([(comment, get_token_count(comment, tokenizer_summ))],
                                            tokenizer_summ, max_tokens)
        summaries = [summarize_text(chunk, tokenizer_summ, model_summ, device, max_length, min_length) for chunk in chunks]
        full_summary = " ".join(summaries)
        while get_token_count(full_summary, tokenizer_summ) > max_length:
            full_summary = summarize_text(full_summary, tokenizer_summ, model_summ, device, max_length, min_length)
        summaries_dict[comment] = full_summary
    st.info("Preprocessing and summarization completed.")
    return summaries_dict

# -------------------------------------------
# Vectorized assignment of categories
def assign_categories_vectorized(batch_embeddings, keyword_embeddings, keyword_keys):
    keyword_matrix = np.stack(list(keyword_embeddings.values()))
    similarity_matrix = cosine_similarity(batch_embeddings, keyword_matrix)
    max_scores = similarity_matrix.max(axis=1)
    max_indices = similarity_matrix.argmax(axis=1)
    assigned = [keyword_keys[idx] for idx in max_indices]
    return max_scores, assigned

# -------------------------------------------
# Process a DataFrame chunk of feedback (with emerging issue clustering)
@st.cache_data(persist="disk")
def process_feedback_data(feedback_data, comment_column, date_column, categories, similarity_threshold):
    global previous_categories
    # Ensure the date column is datetime
    feedback_data[date_column] = pd.to_datetime(feedback_data[date_column], errors='coerce')
    keyword_embeddings = compute_keyword_embeddings(categories)
    if previous_categories != categories:
        keyword_embeddings = compute_keyword_embeddings(categories)
        previous_categories = categories.copy()
    num_rows = len(feedback_data)
    summaries_dict = preprocess_comments_and_summarize(feedback_data, comment_column)
    feedback_data['preprocessed_comments'] = feedback_data[comment_column].apply(preprocess_text)
    feedback_data['Summarized Text'] = feedback_data['preprocessed_comments'].map(summaries_dict)
    feedback_data['Summarized Text'].fillna(feedback_data['preprocessed_comments'], inplace=True)
    batch_size = 1024
    comment_embeddings = []
    for i in range(0, num_rows, batch_size):
        batch = feedback_data['Summarized Text'].iloc[i:i+batch_size].tolist()
        comment_embeddings.extend(model.encode(batch, show_progress_bar=False))
    feedback_data['comment_embeddings'] = comment_embeddings
    feedback_data['Sentiment'] = feedback_data['preprocessed_comments'].apply(perform_sentiment_analysis)
    similarity_scores = np.zeros(num_rows)
    categories_list = [''] * num_rows
    sub_categories_list = [''] * num_rows
    keyphrases_list = [''] * num_rows
    keyword_keys = list(keyword_embeddings.keys())
    for i in range(0, num_rows, batch_size):
        batch_emb = np.array(feedback_data['comment_embeddings'].iloc[i:i+batch_size].tolist())
        if batch_emb.shape[0] == 0:
            continue
        max_scores, assigned_keys = assign_categories_vectorized(batch_emb, keyword_embeddings, keyword_keys)
        for j, key in enumerate(assigned_keys):
            idx = i + j
            score = max_scores[j]
            cat, subcat, kw = key
            similarity_scores[idx] = score
            categories_list[idx] = cat
            sub_categories_list[idx] = subcat
            keyphrases_list[idx] = kw
    # Emerging Issue Clustering if enabled
    if similarity_threshold is not None:
        no_match_indices = [i for i, score in enumerate(similarity_scores) if score < similarity_threshold]
        if len(no_match_indices) > 1:
            st.info(f"Clustering {len(no_match_indices)} comments as emerging issues...")
            no_match_emb = np.array([feedback_data.iloc[i]['comment_embeddings'] for i in no_match_indices])
            num_clusters = min(10, len(no_match_indices))
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(no_match_emb)
            model_summ, tokenizer_summ, device = get_summarization_model_and_tokenizer()
            cluster_labels = {}
            for cluster_id in range(num_clusters):
                cluster_idxs = [no_match_indices[j] for j, c in enumerate(clusters) if c == cluster_id]
                if not cluster_idxs:
                    continue
                cluster_emb = np.array([feedback_data.iloc[i]['comment_embeddings'] for i in cluster_idxs])
                centroid = kmeans.cluster_centers_[cluster_id]
                distances = cosine_similarity([centroid], cluster_emb)[0]
                closest_idx = cluster_idxs[np.argmax(distances)]
                centroid_comment = feedback_data.iloc[closest_idx]['Summarized Text']
                cluster_summary = summarize_text(centroid_comment, tokenizer_summ, model_summ, device, max_length=75, min_length=30)
                cluster_labels[cluster_id] = cluster_summary
            for idx, cluster in zip(no_match_indices, clusters):
                sub_categories_list[idx] = f"Emerging Issue: {cluster_labels[cluster]}"
                categories_list[idx] = "No Match"
    feedback_data.drop(columns=['comment_embeddings'], inplace=True)
    output_rows = []
    for idx, row in feedback_data.iterrows():
        preprocessed = row['preprocessed_comments']
        summarized = row['Summarized Text']
        sentiment = row['Sentiment']
        cat = categories_list[idx]
        subcat = sub_categories_list[idx]
        keyphrase = keyphrases_list[idx]
        best_score = similarity_scores[idx]
        parsed_date = row[date_column].date() if pd.notnull(row[date_column]) else None
        hour = row[date_column].hour if pd.notnull(row[date_column]) else None
        output_rows.append(row.tolist() + [preprocessed, summarized, cat, subcat, keyphrase, sentiment, best_score, parsed_date, hour])
    existing_cols = feedback_data.columns.tolist()
    additional_cols = [comment_column, 'Summarized Text', 'Category', 'Sub-Category', 'Keyphrase', 'Sentiment', 'Best Match Score', 'Parsed Date', 'Hour']
    headers = existing_cols + additional_cols
    trends_data = pd.DataFrame(output_rows, columns=headers)
    trends_data = trends_data.loc[:, ~trends_data.columns.duplicated()]
    return trends_data

# -------------------------------------------
# Streamlit UI setup
st.set_page_config(layout="wide")
st.title("👨‍💻 Transcript Categorization")

# Initialize BERT model
model = initialize_bert_model()

# Sidebar: Emerging Issue Mode & Similarity Threshold
emerging_issue_mode = st.sidebar.checkbox("Emerging Issue Mode")
similarity_threshold = None
if emerging_issue_mode:
    similarity_threshold = st.sidebar.slider("Semantic Similarity Threshold", min_value=0.0, max_value=1.0, value=0.35)
st.sidebar.write("If a comment’s best similarity score is below the threshold, it will be marked as NO MATCH and clustered.")

# Sidebar: Edit Categories
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

# File upload and column selection
uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file is not None:
    csv_data = uploaded_file.read()
    result = chardet.detect(csv_data)
    encoding = result['encoding']
    uploaded_file.seek(0)
    total_rows = sum(1 for row in uploaded_file) - 1  # subtract header
    chunksize = 32  # adjust as needed
    estimated_total_chunks = math.ceil(total_rows / chunksize)
    try:
        first_chunk = next(pd.read_csv(BytesIO(csv_data), encoding=encoding, chunksize=1))
        column_names = first_chunk.columns.tolist()
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
    comment_column = st.selectbox("Select the column containing the comments", column_names)
    date_column = st.selectbox("Select the column containing the dates", column_names)
    grouping_option = st.radio("Select how to group the dates", ["Date", "Week", "Month", "Quarter", "Hour"])
    process_button = st.button("Process Feedback")
    
    # UI placeholders
    progress_bar = st.progress(0)
    trends_dataframe_placeholder = st.empty()
    download_link_placeholder = st.empty()
    st.subheader("All Categories Trends Line Chart")
    line_chart_placeholder = st.empty()
    st.subheader("Pivot Table of Trends")
    pivot_table_placeholder = st.empty()
    st.subheader("Category vs Sentiment and Quantity")
    category_sentiment_dataframe_placeholder = st.empty()
    category_sentiment_bar_chart_placeholder = st.empty()
    st.subheader("Sub-Category vs Sentiment and Quantity")
    subcategory_sentiment_dataframe_placeholder = st.empty()
    subcategory_sentiment_bar_chart_placeholder = st.empty()
    st.subheader("Top 10 Most Recent Comments for Each Top Subcategory")
    combined_placeholders = [(st.empty(), st.empty()) for _ in range(10)]
    
    if process_button:
        processed_chunks = []
        for i, feedback_data in enumerate(pd.read_csv(BytesIO(csv_data), encoding=encoding, chunksize=chunksize)):
            st.info(f"Processing chunk {i+1} of ~{estimated_total_chunks} with {len(feedback_data)} rows.")
            processed_chunk = process_feedback_data(feedback_data, comment_column, date_column, default_categories, similarity_threshold)
            processed_chunks.append(processed_chunk)
            trends_data = pd.concat(processed_chunks, ignore_index=True)
            trends_dataframe_placeholder.dataframe(trends_data)
            progress_bar.progress((i + 1) / estimated_total_chunks)
        # After processing all chunks, build visualizations and export
        st.subheader("Feedback Trends and Insights")
        st.dataframe(trends_data)
        # Ensure 'Parsed Date' is datetime
        trends_data['Parsed Date'] = pd.to_datetime(trends_data['Parsed Date'], errors='coerce')
        # Build pivot table
        if grouping_option == 'Date':
            pivot = trends_data.pivot_table(
                index=['Category', 'Sub-Category'],
                columns=pd.Grouper(key='Parsed Date', freq='D'),
                values='Sentiment',
                aggfunc='count',
                fill_value=0
            )
        elif grouping_option == 'Week':
            pivot = trends_data.pivot_table(
                index=['Category', 'Sub-Category'],
                columns=pd.Grouper(key='Parsed Date', freq='W-SUN', closed='left', label='left'),
                values='Sentiment',
                aggfunc='count',
                fill_value=0
            )
        elif grouping_option == 'Month':
            pivot = trends_data.pivot_table(
                index=['Category', 'Sub-Category'],
                columns=pd.Grouper(key='Parsed Date', freq='M'),
                values='Sentiment',
                aggfunc='count',
                fill_value=0
            )
        elif grouping_option == 'Quarter':
            pivot = trends_data.pivot_table(
                index=['Category', 'Sub-Category'],
                columns=pd.Grouper(key='Parsed Date', freq='Q'),
                values='Sentiment',
                aggfunc='count',
                fill_value=0
            )
        elif grouping_option == 'Hour':
            if 'Hour' not in trends_data.columns:
                feedback_data[date_column] = pd.to_datetime(feedback_data[date_column])
                trends_data['Hour'] = feedback_data[date_column].dt.hour
            pivot = trends_data.pivot_table(
                index=['Category', 'Sub-Category'],
                columns='Hour',
                values='Sentiment',
                aggfunc='count',
                fill_value=0
            )
            pivot.columns = pd.to_datetime(pivot.columns, format='%H').time
        if grouping_option != 'Hour':
            pivot.columns = pivot.columns.strftime('%Y-%m-%d')
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        pivot = pivot[sorted(pivot.columns, reverse=True)]
        pivot_reset = pivot.reset_index().set_index('Sub-Category').drop(columns=['Category'])
        top_5_trends = pivot_reset.head(5).T
        line_chart_placeholder.line_chart(top_5_trends)
        pivot_table_placeholder.dataframe(pivot)
        pivot1 = trends_data.groupby('Category')['Sentiment'].agg(['mean', 'count'])
        pivot1.columns = ['Average Sentiment', 'Quantity']
        pivot1 = pivot1.sort_values('Quantity', ascending=False)
        category_sentiment_bar_chart_placeholder.bar_chart(pivot1['Quantity'])
        category_sentiment_dataframe_placeholder.dataframe(pivot1)
        pivot2 = trends_data.groupby(['Category', 'Sub-Category'])['Sentiment'].agg(['mean', 'count'])
        pivot2.columns = ['Average Sentiment', 'Quantity']
        pivot2 = pivot2.sort_values('Quantity', ascending=False)
        pivot2_reset = pivot2.reset_index().set_index('Sub-Category')
        subcategory_sentiment_bar_chart_placeholder.bar_chart(pivot2_reset['Quantity'])
        subcategory_sentiment_dataframe_placeholder.dataframe(pivot2_reset)
        top_subcategories = pivot2_reset.head(10).index.tolist()
        for idx, subcat in enumerate(top_subcategories):
            title_placeholder, table_placeholder = combined_placeholders[idx]
            title_placeholder.subheader(subcat)
            filtered_data = trends_data[trends_data['Sub-Category'] == subcat]
            top_comments = filtered_data.nlargest(10, 'Parsed Date')[['Parsed Date', comment_column, 'Summarized Text', 'Keyphrase', 'Sentiment', 'Best Match Score']]
            top_comments['Parsed Date'] = top_comments['Parsed Date'].dt.date.astype(str)
            table_placeholder.table(top_comments)
        trends_data['Parsed Date'] = trends_data['Parsed Date'].dt.strftime('%Y-%m-%d').fillna('')
        # Excel export
        excel_file = BytesIO()
        with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
            trends_data.to_excel(writer, sheet_name='Feedback Trends and Insights', index=False)
        excel_file.seek(0)
        b64 = base64.b64encode(excel_file.read()).decode()
        st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="feedback_trends.xlsx">Download Excel File</a>', unsafe_allow_html=True)
