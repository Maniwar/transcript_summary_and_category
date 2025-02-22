import os

# Set the environment variable to control tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # or "false" to enable or disable parallelism

# Now you can import the rest of the modules
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans  # <-- for Emerging Issue clustering
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

class SummarizationDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze()

# Initialize BERT model
@st.cache_resource
def initialize_bert_model():
    start_time = time.time()
    print("Initializing BERT model...")
    end_time = time.time()
    print(f"BERT model initialized. Time taken: {end_time - start_time} seconds.")
    return SentenceTransformer('all-mpnet-base-v2', device="cpu")

# Initialize a variable to store the previous state of the categories
previous_categories = None

# Function to compute keyword embeddings
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

# Function to preprocess the text
def preprocess_text(text):
    if isinstance(text, float):
        text = str(text)
    text = text.encode('ascii', 'ignore').decode(encoding)
    text = re.sub(r'[^\w\s]', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove page breaks
    text = text.replace('\n', ' ').replace('\r', '')
    # Remove non-breaking spaces
    text = text.replace('&nbsp;', ' ')
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to perform sentiment analysis
def perform_sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    return compound_score

# Function to compute the token count of a text
def get_token_count(text, tokenizer):
    return len(tokenizer.encode(text)) - 2

# Function to split comments into chunks
def split_comments_into_chunks(comments, tokenizer, max_tokens):
    sorted_comments = sorted(comments, key=lambda x: x[1], reverse=True)
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0

    for comment, tokens in sorted_comments:
        if tokens > max_tokens:
            # If a single comment exceeds max_tokens, split it
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

@st.cache_resource
def get_summarization_model_and_tokenizer():
    model_name = "knkarthick/MEETING_SUMMARY"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return model, tokenizer, device

def summarize_text(text, tokenizer, model, device, max_length=75, min_length=30):
    input_ids = tokenizer([text], truncation=True, padding=True, return_tensors='pt')['input_ids'].to(device)
    summary_ids = model.generate(input_ids, max_length=max_length, min_length=min_length)[0]
    return tokenizer.decode(summary_ids, skip_special_tokens=True)

def preprocess_comments_and_summarize(
    feedback_data,
    comment_column,
    batch_size=32,
    max_length=75,
    min_length=30,
    max_tokens=1000,
    very_short_limit=30
):
    print("Starting preprocessing and summarization...")

    # 1. Preprocess the comments
    feedback_data['preprocessed_comments'] = feedback_data[comment_column].apply(preprocess_text)
    print("Comments preprocessed.")

    # 2. Get the cached model and tokenizer
    model_summ, tokenizer_summ, device = get_summarization_model_and_tokenizer()
    print("Summarization model and tokenizer retrieved from cache.")

    # 3. Separate comments into categories by token length
    all_comments = feedback_data['preprocessed_comments'].tolist()
    very_short_comments = [c for c in all_comments if get_token_count(c, tokenizer_summ) <= very_short_limit]
    short_comments = [c for c in all_comments if very_short_limit < get_token_count(c, tokenizer_summ) <= max_tokens]
    long_comments = [c for c in all_comments if get_token_count(c, tokenizer_summ) > max_tokens]
    print(f"Separated comments into categories: {len(very_short_comments)} very short, {len(short_comments)} short, {len(long_comments)} long comments.")

    # 4. Summaries dict: store final summaries
    summaries_dict = {c: c for c in very_short_comments}
    print(f"{len(very_short_comments)} very short comments directly added to summaries.")

    # 5. Summarize short comments in batches
    pbar = tqdm(total=len(short_comments), desc="Summarizing short comments")
    for i in range(0, len(short_comments), batch_size):
        batch = short_comments[i : i + batch_size]
        summaries = [summarize_text(c, tokenizer_summ, model_summ, device, max_length, min_length) for c in batch]
        for original_comment, summary in zip(batch, summaries):
            summaries_dict[original_comment] = summary
        pbar.update(len(batch))
    pbar.close()

    # 6. Summarize long comments
    pbar = tqdm(total=len(long_comments), desc="Summarizing long comments")
    for comment in long_comments:
        chunks = split_comments_into_chunks([(comment, get_token_count(comment, tokenizer_summ))], tokenizer_summ, max_tokens)
        chunk_summaries = [
            summarize_text(chunk, tokenizer_summ, model_summ, device, max_length, min_length)
            for chunk in chunks
        ]
        full_summary = " ".join(chunk_summaries)

        # Possibly re-summarize if still too long
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

def compute_semantic_similarity(comment_embedding, keyword_embedding):
    return cosine_similarity([comment_embedding], [keyword_embedding])[0][0]

# Set the default layout mode to "wide"
st.set_page_config(layout="wide")

# Streamlit interface
st.title("👨‍💻 Transcript Categorization")

#Initialize BERT once and cache it
model = initialize_bert_model()

# Add checkbox for emerging issue mode
emerging_issue_mode = st.sidebar.checkbox("Emerging Issue Mode")

# Sidebar description for emerging issue mode
st.sidebar.write(
    "Emerging issue mode allows you to set a minimum similarity score. "
    "If the comment doesn't match up to the categories based on the threshold, "
    "it will be set to NO MATCH (and will be clustered)."
)

# Add slider for semantic similarity threshold in emerging issue mode
similarity_threshold = None
if emerging_issue_mode:
    similarity_threshold = st.sidebar.slider(
        "Semantic Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.35
    )

# Edit categories
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

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

# Placeholders
comment_column = None
date_column = None
trends_data = None
all_processed_data = []
feedback_data = pd.DataFrame()

if uploaded_file is not None:
    csv_data = uploaded_file.read()
    result = chardet.detect(csv_data)
    encoding = result['encoding']

    # Count total rows
    uploaded_file.seek(0)
    total_rows = sum(1 for _ in uploaded_file) - 1

    # Estimate total chunks
    chunksize = 32
    estimated_total_chunks = math.ceil(total_rows / chunksize)

    # Reset file pointer, read first chunk to get columns
    uploaded_file.seek(0)
    try:
        first_chunk = next(pd.read_csv(BytesIO(csv_data), encoding=encoding, chunksize=1))
        column_names = first_chunk.columns.tolist()
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")

    # UI elements for column selection
    comment_column = st.selectbox("Select the column containing the comments", column_names)
    date_column = st.selectbox("Select the column containing the dates", column_names)
    grouping_option = st.radio("Select how to group the dates", ["Date", "Week", "Month", "Quarter", "Hour"])
    process_button = st.button("Process Feedback")

    # UI placeholders
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

    @st.cache_data(persist="disk")
    def process_feedback_data(feedback_data, comment_column, date_column, categories, similarity_threshold):
        global previous_categories
        # 1. Compute or retrieve keyword embeddings
        keyword_embeddings = compute_keyword_embeddings(categories)
        if previous_categories != categories:
            keyword_embeddings = compute_keyword_embeddings(categories)
            previous_categories = categories.copy()
        else:
            if not keyword_embeddings:
                keyword_embeddings = compute_keyword_embeddings(categories)

        # 2. Preprocess & Summarize
        start_time = time.time()
        summaries_dict = preprocess_comments_and_summarize(feedback_data, comment_column)
        feedback_data['preprocessed_comments'] = feedback_data[comment_column].apply(preprocess_text)
        feedback_data['summarized_comments'] = feedback_data['preprocessed_comments'].map(summaries_dict)
        feedback_data['summarized_comments'].fillna(feedback_data['preprocessed_comments'], inplace=True)
        end_time = time.time()
        print(f"Preprocessing + summarization took: {end_time - start_time} seconds.")

        # 3. Compute comment embeddings
        batch_size = 1024
        comment_embeddings = []
        for i in range(0, len(feedback_data), batch_size):
            batch = feedback_data['summarized_comments'][i:i+batch_size].tolist()
            encoded_batch = model.encode(batch, show_progress_bar=False)
            comment_embeddings.extend(encoded_batch)
        feedback_data['comment_embeddings'] = comment_embeddings

        # 4. Compute sentiment scores
        feedback_data['sentiment_scores'] = feedback_data['preprocessed_comments'].apply(perform_sentiment_analysis)

        # 5. Match to categories
        categories_list = [''] * len(feedback_data)
        sub_categories_list = [''] * len(feedback_data)
        keyphrases_list = [''] * len(feedback_data)
        similarity_scores = [0.0] * len(feedback_data)

        # Convert dictionary to list for faster iteration
        kw_keys = list(keyword_embeddings.keys())
        kw_vals = list(keyword_embeddings.values())

        for i in range(0, len(feedback_data), batch_size):
            batch_embs = feedback_data['comment_embeddings'][i : i + batch_size].tolist()
            for j, emb in enumerate(batch_embs):
                idx = i + j
                best_score = 0.0
                best_cat = ""
                best_sub = ""
                best_key = ""
                # Check all keywords
                for (cat, sub, kw), k_emb in zip(kw_keys, kw_vals):
                    score = compute_semantic_similarity(emb, k_emb)
                    if score > best_score:
                        best_score = score
                        best_cat = cat
                        best_sub = sub
                        best_key = kw
                similarity_scores[idx] = best_score
                categories_list[idx] = best_cat
                sub_categories_list[idx] = best_sub
                keyphrases_list[idx] = best_key

        # 6. Emerging issue clustering if threshold is set
        if emerging_issue_mode and similarity_threshold is not None:
            no_match_indices = [i for i, sc in enumerate(similarity_scores) if sc < similarity_threshold]
            if len(no_match_indices) > 1:
                print(f"Clustering {len(no_match_indices)} 'No Match' comments.")
                no_match_embs = np.array([feedback_data.iloc[i]['comment_embeddings'] for i in no_match_indices])
                num_clusters = min(10, len(no_match_indices))
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                clusters = kmeans.fit_predict(no_match_embs)

                # Summarize a representative from each cluster
                model_sum, tokenizer_sum, device = get_summarization_model_and_tokenizer()
                cluster_labels = {}
                for cluster_id in range(num_clusters):
                    cluster_inds = [no_match_indices[k] for k, c in enumerate(clusters) if c == cluster_id]
                    if not cluster_inds:
                        continue
                    cluster_batch = np.array([feedback_data.iloc[ii]['comment_embeddings'] for ii in cluster_inds])
                    centroid = kmeans.cluster_centers_[cluster_id]
                    dists = cosine_similarity([centroid], cluster_batch)[0]
                    best_idx = cluster_inds[np.argmax(dists)]
                    centroid_comment = feedback_data.iloc[best_idx]['summarized_comments']
                    cluster_summary = summarize_text(centroid_comment, tokenizer_sum, model_sum, device, 75, 30)
                    cluster_labels[cluster_id] = cluster_summary

                for idx, cluster_id in zip(no_match_indices, clusters):
                    categories_list[idx] = "No Match"
                    sub_categories_list[idx] = f"Emerging Issue: {cluster_labels[cluster_id]}"

        # Drop embeddings
        feedback_data.drop(columns=['comment_embeddings'], inplace=True)

        # 7. Build final DataFrame
        categorized_comments = []
        for i in range(len(feedback_data)):
            row = feedback_data.iloc[i]
            best_score = similarity_scores[i]
            cat = categories_list[i]
            subcat = sub_categories_list[i]
            keyp = keyphrases_list[i]
            preproc = row['preprocessed_comments']
            summarized = row['summarized_comments']
            sent = row['sentiment_scores']
            parsed_date = row[date_column].split(' ')[0] if isinstance(row[date_column], str) else None
            hr = pd.to_datetime(row[date_column]).hour if pd.notnull(row[date_column]) else None

            row_extended = row.tolist() + [
                preproc,
                summarized,
                cat,
                subcat,
                keyp,
                sent,
                best_score,
                parsed_date,
                hr
            ]
            categorized_comments.append(row_extended)

        existing_cols = feedback_data.columns.tolist()
        additional_cols = [
            comment_column,
            'Summarized Text',
            'Category',
            'Sub-Category',
            'Keyphrase',
            'Sentiment',
            'Best Match Score',
            'Parsed Date',
            'Hour'
        ]
        headers = existing_cols + additional_cols
        trends_data = pd.DataFrame(categorized_comments, columns=headers)

        # Drop duplicated columns
        trends_data = trends_data.loc[:, ~trends_data.columns.duplicated()]
        return trends_data

    if process_button and comment_column and date_column and grouping_option:
        # Start reading the file in chunks
        chunk_iter = pd.read_csv(BytesIO(csv_data), encoding=encoding, chunksize=chunksize)

        processed_chunks = []
        for feedback_data in chunk_iter:
            processed_chunk = process_feedback_data(
                feedback_data, comment_column, date_column,
                default_categories, similarity_threshold
            )
            processed_chunks.append(processed_chunk)

            # Concatenate results
            trends_data = pd.concat(processed_chunks, ignore_index=True)

            # Show updated data
            trends_dataframe_placeholder.dataframe(trends_data)

            processed_chunks_count += 1
            progress_bar.progress(processed_chunks_count / estimated_total_chunks)

            # Once we have the cumulative data, do the same pivot logic as before
            if trends_data is not None:
                trends_data['Parsed Date'] = pd.to_datetime(trends_data['Parsed Date'], errors='coerce')
                # Build pivot
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
                        # Extract hour from date
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

                pivot.columns = pivot.columns.astype(str)
                pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
                pivot = pivot[sorted(pivot.columns, reverse=True)]

                pivot_reset = pivot.reset_index()
                if 'Sub-Category' in pivot_reset.columns:
                    pivot_reset = pivot_reset.set_index('Sub-Category')
                if 'Category' in pivot_reset.columns:
                    pivot_reset = pivot_reset.drop(columns=['Category'], errors='ignore')

                # Top 5 line chart
                top_5_trends = pivot_reset.head(5).T
                line_chart_placeholder.line_chart(top_5_trends)
                pivot_table_placeholder.dataframe(pivot)

                # Additional pivot analyses
                pivot1 = trends_data.groupby('Category')['Sentiment'].agg(['mean', 'count'])
                pivot1.columns = ['Average Sentiment', 'Quantity']
                pivot1 = pivot1.sort_values('Quantity', ascending=False)
                pivot2 = trends_data.groupby(['Category', 'Sub-Category'])['Sentiment'].agg(['mean', 'count'])
                pivot2.columns = ['Average Sentiment', 'Quantity']
                pivot2 = pivot2.sort_values('Quantity', ascending=False)
                pivot2_reset = pivot2.reset_index().set_index('Sub-Category')

                # Show bar charts
                category_sentiment_bar_chart_placeholder.bar_chart(pivot1['Quantity'])
                category_sentiment_dataframe_placeholder.dataframe(pivot1)
                subcategory_sentiment_bar_chart_placeholder.bar_chart(pivot2_reset['Quantity'])
                subcategory_sentiment_dataframe_placeholder.dataframe(pivot2_reset)

                # Show top comments for top subcategories
                top_subcategories = pivot2_reset.head(10).index.tolist()
                for idx, subcat in enumerate(top_subcategories):
                    title_placeholder, table_placeholder = combined_placeholders[idx]
                    title_placeholder.subheader(subcat)
                    filtered_data = trends_data[trends_data['Sub-Category'] == subcat]
                    top_comments = filtered_data.nlargest(10, 'Parsed Date')[[
                        'Parsed Date',
                        comment_column,
                        'Summarized Text',
                        'Keyphrase',
                        'Sentiment',
                        'Best Match Score'
                    ]]
                    top_comments['Parsed Date'] = top_comments['Parsed Date'].dt.date.astype(str)
                    table_placeholder.table(top_comments)

                # Convert date to string
                trends_data['Parsed Date'] = trends_data['Parsed Date'].dt.strftime('%Y-%m-%d').fillna('')
                # Another pivot for final usage
                pivot = trends_data.pivot_table(
                    index=['Category', 'Sub-Category'],
                    columns=pd.to_datetime(trends_data['Parsed Date']).dt.strftime('%Y-%m-%d'),
                    values='Sentiment',
                    aggfunc='count',
                    fill_value=0
                )
                pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
                pivot = pivot[sorted(pivot.columns, reverse=True)]

            # Build final Excel
            excel_file = BytesIO()
            with pd.ExcelWriter(excel_file, engine='xlsxwriter', mode='xlsx') as excel_writer:
                trends_data.to_excel(excel_writer, sheet_name='Feedback Trends and Insights', index=False)
                # Convert 'Parsed Date'
                trends_data['Parsed Date'] = pd.to_datetime(trends_data['Parsed Date'], errors='coerce')
                trends_data['Formatted Date'] = trends_data['Parsed Date'].dt.strftime('%Y-%m-%d')
                # Before resetting, remove level_0 if exists
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
                # If not hour, unify date format
                if grouping_option != 'Hour':
                    pivot.columns = pivot.columns.strftime('%Y-%m-%d')

                pivot.to_excel(excel_writer, sheet_name='Trends by ' + grouping_option, merge_cells=False)
                pivot1.to_excel(excel_writer, sheet_name='Categories', merge_cells=False)
                pivot2.to_excel(excel_writer, sheet_name='Subcategories', merge_cells=False)

                # Example comments
                example_comments_sheet = excel_writer.book.add_worksheet('Example Comments')
                for subcat in top_subcategories:
                    filtered_data = trends_data[trends_data['Sub-Category'] == subcat]
                    top_comments = filtered_data.nlargest(10, 'Parsed Date')[[ 'Parsed Date', comment_column ]]
                    start_row = (top_subcategories.index(subcat) * 8) + 1
                    example_comments_sheet.merge_range(start_row, 0, start_row, 1, subcat)
                    example_comments_sheet.write(start_row, 2, '')
                    example_comments_sheet.write(start_row + 1, 0, 'Date')
                    example_comments_sheet.write(start_row + 1, 1, comment_column)
                    for i, (_, row) in enumerate(top_comments.iterrows(), start=start_row + 2):
                        example_comments_sheet.write(i, 0, str(row['Parsed Date']))
                        example_comments_sheet.write_string(i, 1, str(row[comment_column]))

            if not excel_writer.book.fileclosed:
                excel_writer.close()

            excel_file.seek(0)
            b64 = base64.b64encode(excel_file.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="feedback_trends.xlsx">Download Excel File</a>'
            download_link_placeholder.markdown(href, unsafe_allow_html=True)
