import os

# Set the environment variable to control tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # or "false" to enable or disable parallelism

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN  # <-- DBSCAN for Emerging Issue clustering
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

@st.cache_resource
def initialize_bert_model():
    start_time = time.time()
    print("Initializing BERT model...")
    end_time = time.time()
    print(f"BERT model initialized. Time taken: {end_time - start_time:.2f} seconds.")
    return SentenceTransformer('all-mpnet-base-v2', device="cpu")

previous_categories = None

@st.cache_data(persist="disk")
def compute_keyword_embeddings(categories):
    start_time = time.time()
    print("Computing keyword embeddings...")
    keyword_embeddings = {}
    for category, subcategories in categories.items():
        for subcategory, keywords in subcategories.items():
            for keyword in keywords:
                key = (category, subcategory, keyword)
                if key not in keyword_embeddings:
                    keyword_embeddings[key] = model.encode([keyword])[0]
    end_time = time.time()
    print(f"Keyword embeddings computed. Time taken: {end_time - start_time:.2f} seconds.")
    return keyword_embeddings

def preprocess_text(text):
    if isinstance(text, float):
        text = str(text)
    text = text.encode('ascii', 'ignore').decode(encoding)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.replace('\n', ' ').replace('\r', '')
    text = text.replace('&nbsp;', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def perform_sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores['compound']

def get_token_count(text, tokenizer):
    return len(tokenizer.encode(text)) - 2

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
    feedback_data['preprocessed_comments'] = feedback_data[comment_column].apply(preprocess_text)
    print("Comments preprocessed.")
    model_summ, tokenizer_summ, device = get_summarization_model_and_tokenizer()
    print("Summarization model and tokenizer retrieved from cache.")

    all_comments = feedback_data['preprocessed_comments'].tolist()
    very_short_comments = [c for c in all_comments if get_token_count(c, tokenizer_summ) <= very_short_limit]
    short_comments = [c for c in all_comments if very_short_limit < get_token_count(c, tokenizer_summ) <= max_tokens]
    long_comments = [c for c in all_comments if get_token_count(c, tokenizer_summ) > max_tokens]
    print(f"Separated comments into categories: {len(very_short_comments)} very short, {len(short_comments)} short, {len(long_comments)} long comments.")

    summaries_dict = {c: c for c in very_short_comments}
    print(f"{len(very_short_comments)} very short comments directly added to summaries.")

    pbar = tqdm(total=len(short_comments), desc="Summarizing short comments")
    for i in range(0, len(short_comments), batch_size):
        batch = short_comments[i:i+batch_size]
        summaries = [summarize_text(c, tokenizer_summ, model_summ, device, max_length, min_length) for c in batch]
        for original_comment, summary in zip(batch, summaries):
            summaries_dict[original_comment] = summary
        pbar.update(len(batch))
    pbar.close()

    pbar = tqdm(total=len(long_comments), desc="Summarizing long comments")
    for comment in long_comments:
        chunks = split_comments_into_chunks([(comment, get_token_count(comment, tokenizer_summ))], tokenizer_summ, max_tokens)
        chunk_summaries = [summarize_text(chunk, tokenizer_summ, model_summ, device, max_length, min_length) for chunk in chunks]
        full_summary = " ".join(chunk_summaries)
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

    summaries_dict = preprocess_comments_and_summarize(feedback_data, comment_column)
    feedback_data['preprocessed_comments'] = feedback_data[comment_column].apply(preprocess_text)
    feedback_data['summarized_comments'] = feedback_data['preprocessed_comments'].map(summaries_dict)
    feedback_data['summarized_comments'] = feedback_data['summarized_comments'].fillna(feedback_data['preprocessed_comments'])

    batch_size = 1024
    comment_embeddings = []
    for i in range(0, len(feedback_data), batch_size):
        batch = feedback_data['summarized_comments'][i:i+batch_size].tolist()
        emb = model.encode(batch, show_progress_bar=False)
        comment_embeddings.extend(emb)
    feedback_data['comment_embeddings'] = comment_embeddings

    feedback_data['sentiment_scores'] = feedback_data['preprocessed_comments'].apply(perform_sentiment_analysis)

    categories_list = [''] * len(feedback_data)
    sub_categories_list = [''] * len(feedback_data)
    keyphrases_list = [''] * len(feedback_data)
    similarity_scores = [0.0] * len(feedback_data)

    kw_keys = list(keyword_embeddings.keys())
    kw_vals = list(keyword_embeddings.values())

    # 1) Assign best categories first by threshold
    for i in range(0, len(feedback_data), batch_size):
        embs_batch = feedback_data['comment_embeddings'][i : i + batch_size].tolist()
        for j, emb in enumerate(embs_batch):
            idx = i + j
            best_score = 0.0
            best_cat = ""
            best_sub = ""
            best_kw = ""
            for (cat, sub, kw), kv in zip(kw_keys, kw_vals):
                score = compute_semantic_similarity(emb, kv)
                if score > best_score:
                    best_score = score
                    best_cat = cat
                    best_sub = sub
                    best_kw = kw
            categories_list[idx] = best_cat
            sub_categories_list[idx] = best_sub
            keyphrases_list[idx] = best_kw
            similarity_scores[idx] = best_score

    # 2) If below threshold => 'No Match', then DBSCAN cluster them
    if emerging_issue_mode and similarity_threshold is not None:
        no_match_indices = [ix for ix, sc in enumerate(similarity_scores) if sc < similarity_threshold]
        if len(no_match_indices) > 1:
            # DBSCAN with metric='cosine'
            from sklearn.cluster import DBSCAN

            no_match_embs = np.array([feedback_data.iloc[ix]['comment_embeddings'] for ix in no_match_indices])
            # example hyperparams: eps=0.7, min_samples=3, metric='cosine'
            dbscan = DBSCAN(eps=0.7, min_samples=3, metric='cosine')
            clusters = dbscan.fit_predict(no_match_embs)

            model_sum, tokenizer_sum, device_sum = get_summarization_model_and_tokenizer()
            cluster_labels = {}

            # group points by cluster label
            cluster_map = defaultdict(list)
            for local_idx, c_id in enumerate(clusters):
                global_idx = no_match_indices[local_idx]
                cluster_map[c_id].append(global_idx)

            # Summarize each cluster c_id != -1
            for c_id, idx_list in cluster_map.items():
                if c_id == -1:
                    continue  # noise => skip
                cluster_vectors = np.array([feedback_data.iloc[ii]['comment_embeddings'] for ii in idx_list])
                centroid = cluster_vectors.mean(axis=0)
                dists = cosine_similarity([centroid], cluster_vectors)[0]
                best_local = np.argmax(dists)
                best_idx = idx_list[best_local]
                centroid_comment = feedback_data.iloc[best_idx]['summarized_comments']
                cluster_summary = summarize_text(centroid_comment, tokenizer_sum, model_sum, device_sum, 75, 30)
                cluster_labels[c_id] = cluster_summary

            # Now assign
            for local_idx, c_id in enumerate(clusters):
                global_idx = no_match_indices[local_idx]
                if c_id == -1:
                    # noise => remain No Match
                    categories_list[global_idx] = "No Match"
                    sub_categories_list[global_idx] = "No Match"
                    keyphrases_list[global_idx] = "No Match"
                else:
                    categories_list[global_idx] = "Emerging Issues"
                    sub_categories_list[global_idx] = f"DBSCAN Cluster: {cluster_labels[c_id]}"
                    keyphrases_list[global_idx] = "DBSCAN"

    feedback_data.drop(columns=['comment_embeddings'], inplace=True)

    rows_extended = []
    for idx in range(len(feedback_data)):
        row = feedback_data.iloc[idx]
        cat = categories_list[idx]
        s_cat = sub_categories_list[idx]
        kwp = keyphrases_list[idx]
        sc = similarity_scores[idx]
        preproc = row['preprocessed_comments']
        sumtext = row['summarized_comments']
        sent = row['sentiment_scores']
        parsed_date = row[date_column].split(' ')[0] if isinstance(row[date_column], str) else None
        hour = pd.to_datetime(row[date_column]).hour if pd.notnull(row[date_column]) else None

        row_ext = row.tolist() + [
            preproc,
            sumtext,
            cat,
            s_cat,
            kwp,
            sent,
            sc,
            parsed_date,
            hour
        ]
        rows_extended.append(row_ext)

    ex_cols = feedback_data.columns.tolist()
    add_cols = [
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
    headers = ex_cols + add_cols
    trends = pd.DataFrame(rows_extended, columns=headers)
    trends = trends.loc[:, ~trends.columns.duplicated()]
    return trends

if __name__ == "__main__":
    st.title("👨‍💻 Transcript Categorization")
    model = initialize_bert_model()

    emerging_issue_mode = st.sidebar.checkbox("Emerging Issue Mode")
    similarity_threshold = None
    if emerging_issue_mode:
        similarity_threshold = st.sidebar.slider("Semantic Similarity Threshold", 0.0, 1.0, 0.35)

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

    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is not None:
        csv_data = uploaded_file.read()
        result = chardet.detect(csv_data)
        encoding = result['encoding']

        uploaded_file.seek(0)
        total_rows = sum(1 for _ in uploaded_file) - 1
        chunksize = 32
        estimated_total_chunks = math.ceil(total_rows / chunksize)

        uploaded_file.seek(0)
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

        if process_button and comment_column and date_column and grouping_option:
            chunk_iter = pd.read_csv(BytesIO(csv_data), encoding=encoding, chunksize=chunksize)
            processed_chunks = []

            for feedback_data in chunk_iter:
                processed_chunk = process_feedback_data(feedback_data, comment_column, date_column, default_categories, similarity_threshold)
                processed_chunks.append(processed_chunk)

                trends_data = pd.concat(processed_chunks, ignore_index=True)
                trends_dataframe_placeholder.dataframe(trends_data)
                processed_chunks_count += 1
                progress_bar.progress(processed_chunks_count / estimated_total_chunks)

                if trends_data is not None:
                    trends_data['Parsed Date'] = pd.to_datetime(trends_data['Parsed Date'], errors='coerce')

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

                    pivot.columns = pivot.columns.astype(str)
                    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
                    pivot = pivot[sorted(pivot.columns, reverse=True)]

                    pivot_reset = pivot.reset_index()
                    if 'Sub-Category' in pivot_reset.columns:
                        pivot_reset = pivot_reset.set_index('Sub-Category')
                    if 'Category' in pivot_reset.columns:
                        pivot_reset = pivot_reset.drop(columns=['Category'], errors='ignore')

                    top_5_trends = pivot_reset.head(5).T
                    line_chart_placeholder.line_chart(top_5_trends)
                    pivot_table_placeholder.dataframe(pivot)

                    pivot1 = trends_data.groupby('Category')['Sentiment'].agg(['mean', 'count'])
                    pivot1.columns = ['Average Sentiment', 'Quantity']
                    pivot1 = pivot1.sort_values('Quantity', ascending=False)

                    pivot2 = trends_data.groupby(['Category', 'Sub-Category'])['Sentiment'].agg(['mean', 'count'])
                    pivot2.columns = ['Average Sentiment', 'Quantity']
                    pivot2 = pivot2.sort_values('Quantity', ascending=False)
                    pivot2_reset = pivot2.reset_index().set_index('Sub-Category')

                    category_sentiment_bar_chart_placeholder.bar_chart(pivot1['Quantity'])
                    category_sentiment_dataframe_placeholder.dataframe(pivot1)
                    subcategory_sentiment_bar_chart_placeholder.bar_chart(pivot2_reset['Quantity'])
                    subcategory_sentiment_dataframe_placeholder.dataframe(pivot2_reset)

                    top_subcategories = pivot2_reset.head(10).index.tolist()
                    for idx, subcat in enumerate(top_subcategories):
                        title_placeholder, table_placeholder = combined_placeholders[idx]
                        title_placeholder.subheader(subcat)
                        filtered_data = trends_data[trends_data['Sub-Category'] == subcat]
                        top_comments = filtered_data.nlargest(10, 'Parsed Date')[
                            ['Parsed Date', comment_column, 'Summarized Text', 'Keyphrase', 'Sentiment', 'Best Match Score']
                        ]
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

                excel_file = BytesIO()
                with pd.ExcelWriter(excel_file, engine='xlsxwriter', mode='xlsx') as excel_writer:
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
                    if grouping_option != 'Hour':
                        pivot.columns = pivot.columns.strftime('%Y-%m-%d')

                    pivot.to_excel(excel_writer, sheet_name='Trends by ' + grouping_option, merge_cells=False)
                    pivot1.to_excel(excel_writer, sheet_name='Categories', merge_cells=False)
                    pivot2.to_excel(excel_writer, sheet_name='Subcategories', merge_cells=False)

                    example_comments_sheet = excel_writer.book.add_worksheet('Example Comments')
                    for subcat in top_subcategories:
                        filtered_data = trends_data[trends_data['Sub-Category'] == subcat]
                        top_comments = filtered_data.nlargest(10, 'Parsed Date')[
                            ['Parsed Date', comment_column]
                        ]
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
