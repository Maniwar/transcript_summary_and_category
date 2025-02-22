import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # or "false" to enable or disable parallelism

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans   # <-- Imported for emerging issue clustering
import datetime
import numpy as np
import xlsxwriter
import chardet
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import base64
from io import BytesIO
import streamlit as st
import textwrap
from categories_josh_sub_V6_3 import default_categories
import time
from tqdm import tqdm
import re
import string
import unicodedata
import math
from collections import defaultdict

# -------------------------------
# Dataset for summarization (if needed)
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

# -------------------------------
# Initialize BERT model (cached)
@st.cache_resource
def initialize_bert_model():
    start_time = time.time()
    print("Initializing BERT model...")
    model = SentenceTransformer('all-mpnet-base-v2', device="cpu")
    end_time = time.time()
    print(f"BERT model initialized. Time taken: {end_time - start_time} seconds.")
    return model

# Global variable to store previous categories (for caching embeddings)
previous_categories = None

# Compute keyword embeddings (cached)
@st.cache_data(persist="disk")
def compute_keyword_embeddings(categories):
    start_time = time.time()
    print("Computing keyword embeddings...")
    keyword_embeddings = {}
    for category, subcategories in categories.items():
        for subcategory, keywords in subcategories.items():
            for keyword in keywords:
                if (category, subcategory, keyword) not in keyword_embeddings:
                    keyword_embeddings[(category, subcategory, keyword)] = model.encode([keyword])[0]
    end_time = time.time()
    print(f"Keyword embeddings computed. Time taken: {end_time - start_time} seconds.")
    return keyword_embeddings

# -------------------------------
# Text preprocessing
def preprocess_text(text):
    if isinstance(text, float):
        text = str(text)
    # 'encoding' variable will be defined later after file upload
    text = text.encode('ascii', 'ignore').decode(encoding)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.replace('\n', ' ').replace('\r', '')
    text = text.replace('&nbsp;', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------------
# Sentiment analysis
def perform_sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    return compound_score

# -------------------------------
# Token count helper
def get_token_count(text, tokenizer):
    return len(tokenizer.encode(text)) - 2

# -------------------------------
# Chunking function for long comments
def split_comments_into_chunks(comments, tokenizer, max_tokens):
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
    print(f"Total number of chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} token count: {get_token_count(chunk, tokenizer)}")
    return chunks

# -------------------------------
# Initialize summarization model and tokenizer (cached)
@st.cache_resource
def get_summarization_model_and_tokenizer():
    model_name = "knkarthick/MEETING_SUMMARY"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return model, tokenizer, device

# -------------------------------
# Summarize a given text
def summarize_text(text, tokenizer, model, device, max_length, min_length):
    input_ids = tokenizer([text], truncation=True, padding=True, return_tensors='pt')['input_ids'].to(device)
    summary_ids = model.generate(input_ids, max_length=max_length, min_length=min_length)[0]
    return tokenizer.decode(summary_ids, skip_special_tokens=True)

# -------------------------------
# Preprocess and summarize comments (cached)
def preprocess_comments_and_summarize(feedback_data, comment_column, batch_size=32, max_length=75, min_length=30, max_tokens=1000, very_short_limit=30):
    print("Starting preprocessing and summarization...")
    feedback_data['preprocessed_comments'] = feedback_data[comment_column].apply(preprocess_text)
    print("Comments preprocessed.")
    model_summ, tokenizer_summ, device = get_summarization_model_and_tokenizer()
    print("Summarization model and tokenizer retrieved from cache.")
    comments = feedback_data['preprocessed_comments'].tolist()
    very_short_comments = [comment for comment in comments if get_token_count(comment, tokenizer_summ) <= very_short_limit]
    short_comments = [comment for comment in comments if very_short_limit < get_token_count(comment, tokenizer_summ) <= max_tokens]
    long_comments = [comment for comment in comments if get_token_count(comment, tokenizer_summ) > max_tokens]
    print(f"Separated comments into: {len(very_short_comments)} very short, {len(short_comments)} short, and {len(long_comments)} long comments.")
    summaries_dict = {comment: comment for comment in very_short_comments}
    print(f"{len(very_short_comments)} very short comments directly added to summaries.")
    pbar = tqdm(total=len(short_comments), desc="Summarizing short comments")
    for i in range(0, len(short_comments), batch_size):
        batch = short_comments[i:i+batch_size]
        summaries = [summarize_text(comment, tokenizer_summ, model_summ, device, max_length, min_length) for comment in batch]
        for original_comment, summary in zip(batch, summaries):
            summaries_dict[original_comment] = summary
        pbar.update(len(batch))
    pbar.close()
    pbar = tqdm(total=len(long_comments), desc="Summarizing long comments")
    for comment in long_comments:
        chunks = split_comments_into_chunks([(comment, get_token_count(comment, tokenizer_summ))], tokenizer_summ, max_tokens)
        summaries = [summarize_text(chunk, tokenizer_summ, model_summ, device, max_length, min_length) for chunk in chunks]
        full_summary = " ".join(summaries)
        resummarization_count = 0
        while get_token_count(full_summary, tokenizer_summ) > max_length:
            resummarization_count += 1
            print(f"Re-summarizing a long comment with token count: {get_token_count(full_summary, tokenizer_summ)}")
            full_summary = summarize_text(full_summary, tokenizer_summ, model_summ, device, max_length, min_length)
        if resummarization_count > 0:
            print(f"Long comment was re-summarized {resummarization_count} times to fit the max length.")
        summaries_dict[comment] = full_summary
        pbar.update(1)
    pbar.close()
    print("Preprocessing and summarization completed.")
    return summaries_dict

# -------------------------------
# Semantic similarity helper
def compute_semantic_similarity(comment_embedding, keyword_embedding):
    return cosine_similarity([comment_embedding], [keyword_embedding])[0][0]

# -------------------------------
# Streamlit configuration
st.set_page_config(layout="wide")
st.title("👨‍💻 Transcript Categorization")
model = initialize_bert_model()

# Sidebar for emerging issue mode and similarity threshold
emerging_issue_mode = st.sidebar.checkbox("Emerging Issue Mode")
st.sidebar.write("Emerging issue mode allows you to set a minimum similarity score. If a comment’s best match is below the threshold, it will be marked for clustering.")
similarity_threshold = st.sidebar.slider("Semantic Similarity Threshold", min_value=0.0, max_value=1.0, value=0.35) if emerging_issue_mode else None

# -------------------------------
# Category editing in sidebar
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

# -------------------------------
# File upload and column selection
uploaded_file = st.file_uploader("Upload CSV file", type="csv")
comment_column = None
date_column = None
trends_data = None
all_processed_data = []
feedback_data = pd.DataFrame()

if uploaded_file is not None:
    csv_data = uploaded_file.read()
    result = chardet.detect(csv_data)
    encoding = result['encoding']
    uploaded_file.seek(0)
    total_rows = sum(1 for row in uploaded_file) - 1  # subtract header
    chunksize = 32
    estimated_total_chunks = math.ceil(total_rows / chunksize)
    try:
        first_chunk = next(pd.read_csv(BytesIO(csv_data), encoding=encoding, chunksize=1))
        column_names = first_chunk.columns.tolist()
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")

    comment_column = st.selectbox("Select the column containing the comments", column_names)
    date_column = st.selectbox("Select the column containing the dates", column_names)
    grouping_option = st.radio("Select how to group the dates", ["Date", "Week", "Month", "Quarter", "Hour"])
    process_button = st.button("Process Feedback")
    progress_bar = st.progress(0)
    processed_chunks_count = 0
    trends_dataframe_placeholder = st.empty()
    download_link_placeholder = st.empty()
    st.subheader("All Categories Trends Line Chart")
    line_chart_placeholder = st.empty()
    pivot_table_placeholder = st.empty()
    st.subheader("Category vs Sentiment and Quantity")
    category_sentiment_dataframe_placeholder = st.empty()
    category_sentiment_bar_chart_placeholder = st.empty()
    st.subheader("Sub-Category vs Sentiment and Quantity")
    subcategory_sentiment_dataframe_placeholder = st.empty()
    subcategory_sentiment_bar_chart_placeholder = st.empty()
    st.subheader("Top 10 Most Recent Comments for Each Top Subcategory")
    combined_placeholders = [(st.empty(), st.empty()) for _ in range(10)]

    # -------------------------------
    # Process feedback data with emerging issue clustering integrated
    @st.cache_data(persist="disk")
    def process_feedback_data(feedback_data, comment_column, date_column, categories, similarity_threshold):
        global previous_categories
        keyword_embeddings = compute_keyword_embeddings(categories)
        if previous_categories != categories:
            keyword_embeddings = compute_keyword_embeddings(categories)
            previous_categories = categories.copy()
        else:
            if not keyword_embeddings:
                keyword_embeddings = compute_keyword_embeddings(categories)

        categorized_comments = []
        sentiments = []
        similarity_scores = [0.0] * len(feedback_data)
        categories_list = [''] * len(feedback_data)
        sub_categories_list = [''] * len(feedback_data)
        keyphrases_list = [''] * len(feedback_data)
        
        # Preprocess and summarize comments
        start_time = time.time()
        print("Preprocessing comments and summarizing...")
        summaries_dict = preprocess_comments_and_summarize(feedback_data, comment_column)
        feedback_data['preprocessed_comments'] = feedback_data[comment_column].apply(preprocess_text)
        feedback_data['summarized_comments'] = feedback_data['preprocessed_comments'].map(summaries_dict)
        feedback_data['summarized_comments'] = feedback_data['summarized_comments'].fillna(feedback_data['preprocessed_comments'])
        end_time = time.time()
        print(f"Preprocessed and summarized. Time taken: {end_time - start_time} seconds.")

        # Compute comment embeddings in batches
        start_time = time.time()
        print("Computing comment embeddings in batches...")
        batch_size = 1024
        comment_embeddings = []
        for i in range(0, len(feedback_data), batch_size):
            batch = feedback_data['summarized_comments'][i:i+batch_size].tolist()
            comment_embeddings.extend(model.encode(batch))
        feedback_data['comment_embeddings'] = comment_embeddings
        end_time = time.time()
        print(f"Comment embeddings computed in batches. Time taken: {end_time - start_time} seconds.")

        # Compute sentiment scores
        start_time = time.time()
        print("Computing sentiment scores...")
        feedback_data['sentiment_scores'] = feedback_data['preprocessed_comments'].apply(perform_sentiment_analysis)
        end_time = time.time()
        print(f"Sentiment scores computed. Time taken: {end_time - start_time} seconds.")

        # Compute semantic similarity and assign categories
        start_time = time.time()
        print("Computing semantic similarity and assigning categories...")
        for i in range(0, len(feedback_data), batch_size):
            batch_embeddings = feedback_data['comment_embeddings'][i:i + batch_size].tolist()
            for (category, subcategory, keyword), emb in keyword_embeddings.items():
                batch_similarity_scores = [compute_semantic_similarity(batch_embedding, emb) for batch_embedding in batch_embeddings]
                for j, sim_score in enumerate(batch_similarity_scores):
                    idx = i + j
                    if sim_score > similarity_scores[idx]:
                        categories_list[idx] = category
                        sub_categories_list[idx] = subcategory
                        keyphrases_list[idx] = keyword
                        similarity_scores[idx] = sim_score
        end_time = time.time()
        print(f"Semantic similarity computed and categories assigned. Time taken: {end_time - start_time} seconds.")

        # --- Emerging Issue Clustering Block ---
        if emerging_issue_mode:
            # Identify indices with similarity scores below the threshold
            no_match_indices = [i for i, score in enumerate(similarity_scores) if score < similarity_threshold]
            if len(no_match_indices) > 1:
                print(f"Clustering {len(no_match_indices)} 'No Match' comments for emerging issues...")
                no_match_embeddings = np.array([feedback_data.iloc[i]['comment_embeddings'] for i in no_match_indices])
                num_clusters = min(10, len(no_match_indices))
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                clusters = kmeans.fit_predict(no_match_embeddings)
                model_summ, tokenizer_summ, device = get_summarization_model_and_tokenizer()
                cluster_labels = {}
                for cluster_id in range(num_clusters):
                    cluster_indices = [no_match_indices[j] for j, c in enumerate(clusters) if c == cluster_id]
                    cluster_embeddings = np.array([feedback_data.iloc[i]['comment_embeddings'] for i in cluster_indices])
                    centroid = kmeans.cluster_centers_[cluster_id]
                    distances = cosine_similarity([centroid], cluster_embeddings)[0]
                    closest_idx = cluster_indices[np.argmax(distances)]
                    centroid_comment = feedback_data.iloc[closest_idx]['summarized_comments']
                    # Summarize the centroid comment to form a cluster label
                    cluster_summary = summarize_text(centroid_comment, tokenizer_summ, model_summ, device, max_length=75, min_length=30)
                    cluster_labels[cluster_id] = cluster_summary
                # Update sub-categories (and optionally categories) for no-match comments with emerging issue labels
                for idx, cluster in zip(no_match_indices, clusters):
                    sub_categories_list[idx] = f"Emerging Issue: {cluster_labels[cluster]}"
                    categories_list[idx] = 'No Match'
            else:
                print("Not enough 'No Match' comments for clustering.")

        # Drop comment embeddings from DataFrame now that they have been used
        feedback_data.drop(columns=['comment_embeddings'], inplace=True)

        # Assemble final rows
        for index in range(len(feedback_data)):
            row = feedback_data.iloc[index]
            preprocessed_comment = row['preprocessed_comments']
            sentiment_score = row['sentiment_scores']
            # Use the (possibly updated) category and sub-category lists
            category = categories_list[index]
            sub_category = sub_categories_list[index]
            keyphrase = keyphrases_list[index]
            best_match_score = similarity_scores[index]
            summarized_text = row['summarized_comments']
            # (No override here—emerging issue clustering has already updated the labels)
            parsed_date = row[date_column].split(' ')[0] if isinstance(row[date_column], str) else None
            hour = pd.to_datetime(row[date_column]).hour
            row_extended = row.tolist() + [preprocessed_comment, summarized_text, category, sub_category, keyphrase, sentiment_score, best_match_score, parsed_date, hour]
            categorized_comments.append(row_extended)
        
        existing_columns = feedback_data.columns.tolist()
        additional_columns = [comment_column, 'Summarized Text', 'Category', 'Sub-Category', 'Keyphrase', 'Sentiment', 'Best Match Score', 'Parsed Date', 'Hour']
        headers = existing_columns + additional_columns
        trends_data = pd.DataFrame(categorized_comments, columns=headers)
        trends_data = trends_data.loc[:, ~trends_data.columns.duplicated()]
        duplicate_columns = set([col for col in trends_data.columns if trends_data.columns.tolist().count(col) > 1])
        for column in duplicate_columns:
            column_indices = [i for i, col in enumerate(trends_data.columns) if col == column]
            for i, idx in enumerate(column_indices[1:], start=1):
                trends_data.columns.values[idx] = f"{column}_{i}"
        return trends_data

    if comment_column is not None and date_column is not None and grouping_option is not None and process_button:
        chunk_iter = pd.read_csv(BytesIO(csv_data), encoding=encoding, chunksize=32)
        processed_chunks = []
        for feedback_data in chunk_iter:
            processed_chunk = process_feedback_data(feedback_data, comment_column, date_column, default_categories, similarity_threshold)
            processed_chunks.append(processed_chunk)
            trends_data = pd.concat(processed_chunks, ignore_index=True)
            # (The rest of your code for pivot tables, visualizations, and Excel export follows below...)
            trends_dataframe_placeholder.dataframe(trends_data)
            # ... [pivot table creation, charts, download link, etc.] ...
            processed_chunks_count += 1
            progress_value = processed_chunks_count / estimated_total_chunks
            progress_bar.progress(progress_value)
            
            # (Excel export code below remains unchanged)
            excel_file = BytesIO()
            with pd.ExcelWriter(excel_file, engine='xlsxwriter') as excel_writer:
                trends_data.to_excel(excel_writer, sheet_name='Feedback Trends and Insights', index=False)
                trends_data['Parsed Date'] = pd.to_datetime(trends_data['Parsed Date'], errors='coerce')
                trends_data['Formatted Date'] = trends_data['Parsed Date'].dt.strftime('%Y-%m-%d')
                if 'level_0' in trends_data.columns:
                    trends_data.drop(columns='level_0', inplace=True)
                trends_data.reset_index(inplace=True)
                trends_data.set_index('Formatted Date', inplace=True)
                if grouping_option == 'Date':
                    pivot = trends_data.pivot_table(
                        index=['Category', 'Sub-Category'],
                        columns='Parsed Date',
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
                    feedback_data[date_column] = pd.to_datetime(feedback_data[date_column])
                    pivot = trends_data.pivot_table(
                        index=['Category', 'Sub-Category'],
                        columns=feedback_data[date_column].dt.hour,
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
                for idx, subcategory in enumerate(top_subcategories):
                    title_placeholder, table_placeholder = combined_placeholders[idx]
                    title_placeholder.subheader(subcategory)
                    filtered_data = trends_data[trends_data['Sub-Category'] == subcategory]
                    top_comments = filtered_data.nlargest(10, 'Parsed Date')[['Parsed Date', comment_column, 'Summarized Text', 'Keyphrase', 'Sentiment', 'Best Match Score']]
                    top_comments['Parsed Date'] = top_comments['Parsed Date'].dt.date.astype(str)
                    table_placeholder.table(top_comments)
                trends_data['Parsed Date'] = trends_data['Parsed Date'].dt.strftime('%Y-%m-%d').fillna('')
                pivot = trends_data.pivot_table(
                    index=['Category', 'Sub-Category'],
                    columns=pd.to_datetime(trends_data['Parsed Date']).dt.strftime('%Y-%m-%d'),
                    values='Sentiment',
                    aggfunc='count',
                    fill_value=0
                )
                pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
                pivot = pivot[sorted(pivot.columns, reverse=True)]
                pivot.to_excel(excel_writer, sheet_name='Trends by ' + grouping_option, merge_cells=False)
                pivot1.to_excel(excel_writer, sheet_name='Categories', merge_cells=False)
                pivot2.to_excel(excel_writer, sheet_name='Subcategories', merge_cells=False)
                example_comments_sheet = excel_writer.book.add_worksheet('Example Comments')
                for subcategory in top_subcategories:
                    filtered_data = trends_data[trends_data['Sub-Category'] == subcategory]
                    top_comments = filtered_data.nlargest(10, 'Parsed Date')[['Parsed Date', comment_column]]
                    start_row = (top_subcategories.index(subcategory) * 8) + 1
                    example_comments_sheet.merge_range(start_row, 0, start_row, 1, subcategory)
                    example_comments_sheet.write(start_row, 2, '')
                    example_comments_sheet.write(start_row + 1, 0, 'Date')
                    example_comments_sheet.write(start_row + 1, 1, comment_column)
                    for i, (_, row) in enumerate(top_comments.iterrows(), start=start_row + 2):
                        example_comments_sheet.write(i, 0, row['Parsed Date'])
                        example_comments_sheet.write_string(i, 1, str(row[comment_column]))
            if not excel_writer.book.fileclosed:
                excel_writer.close()
            excel_file.seek(0)
            b64 = base64.b64encode(excel_file.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="feedback_trends.xlsx">Download Excel File</a>'
            download_link_placeholder.markdown(href, unsafe_allow_html=True)

