import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
import datetime
import numpy as np
import xlsxwriter
import chardet
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import base64
from io import BytesIO
import streamlit as st
import textwrap
from categories_josh_sub_V6_3 import default_categories
import time
from tqdm import tqdm
import re
import math
from collections import defaultdict
from bertopic import BERTopic  # Import BERTopic for emerging issues
from umap import UMAP  # Needed for custom n_neighbors in BERTopic

################################
#    Summarization Dataset     #
################################
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

##################################################
#             Initialization & Globals           #
##################################################
@st.cache_resource
def initialize_bert_model():
    start_time = time.time()
    print("Initializing BERT model...")
    model_ = SentenceTransformer('all-mpnet-base-v2', device="cpu")
    end_time = time.time()
    print(f"BERT model initialized. Time taken: {end_time - start_time:.2f} seconds.")
    return model_

model = None  # Global reference
previous_categories = None

@st.cache_data(persist="disk")
def compute_keyword_embeddings(categories):
    start_time = time.time()
    print("Computing keyword embeddings...")
    keyword_embeddings = {}
    for cat, subcats in categories.items():
        for subcat, keywords in subcats.items():
            for kw in keywords:
                if (cat, subcat, kw) not in keyword_embeddings:
                    keyword_embeddings[(cat, subcat, kw)] = model.encode([kw])[0]
    end_time = time.time()
    print(f"Keyword embeddings computed. Time taken: {end_time - start_time:.2f} seconds.")
    return keyword_embeddings

def preprocess_text(text):
    if isinstance(text, float):
        text = str(text)
    text = text.encode('ascii', 'ignore').decode('ascii')  # Fixed encoding issue
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.replace('\n', ' ').replace('\r', '').replace('\u00a0', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def perform_sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)['compound']

def get_token_count(text, tokenizer):
    return len(tokenizer.encode(text)) - 2

def split_comments_into_chunks(comments, tokenizer, max_tokens):
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

def compute_semantic_similarity(comment_embedding, keyword_embedding):
    return cosine_similarity([comment_embedding], [keyword_embedding])[0][0]

###################################################################
#                 Preprocess & Summarize Comments                #
###################################################################
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
    print(f"Separated comments: {len(very_short_comments)} very short, {len(short_comments)} short, {len(long_comments)} long.")

    summaries_dict = {c: c for c in very_short_comments}
    print(f"{len(very_short_comments)} very short comments directly added to summaries.")

    pbar = tqdm(total=len(short_comments), desc="Summarizing short comments")
    for i in range(0, len(short_comments), batch_size):
        batch = short_comments[i:i+batch_size]
        summaries = [summarize_text(c, tokenizer_summ, model_summ, device, max_length, min_length) for c in batch]
        for oc, summ in zip(batch, summaries):
            summaries_dict[oc] = summ
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
            full_summary = summarize_text(full_summary, tokenizer_summ, model_summ, device, max_length, min_length)
        if resummarization_count > 0:
            print(f"Long comment re-summarized {resummarization_count} times.")
        summaries_dict[comment] = full_summary
        pbar.update(1)
    pbar.close()

    print("Preprocessing and summarization completed.")
    return summaries_dict

###################################################################
#               Chunk-by-Chunk Known Category Assignment          #
###################################################################
def process_feedback_data_chunk(
    feedback_data,
    comment_column,
    date_column,
    categories,
    similarity_threshold
):
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
    feedback_data['summarized_comments'] = feedback_data['preprocessed_comments'].map(summaries_dict).fillna(feedback_data['preprocessed_comments'])

    batch_size = 1024
    comment_embeddings = []
    for i in range(0, len(feedback_data), batch_size):
        batch = feedback_data['summarized_comments'][i:i+batch_size].tolist()
        comment_embeddings.extend(model.encode(batch, show_progress_bar=False))
    feedback_data['comment_embeddings'] = comment_embeddings

    feedback_data['sentiment_scores'] = feedback_data['preprocessed_comments'].apply(perform_sentiment_analysis)

    categories_list = [''] * len(feedback_data)
    sub_categories_list = [''] * len(feedback_data)
    keyphrases_list = [''] * len(feedback_data)
    best_scores = [0.0] * len(feedback_data)

    kw_keys = list(keyword_embeddings.keys())
    kw_vals = list(keyword_embeddings.values())

    for i in range(0, len(feedback_data), batch_size):
        embs_batch = feedback_data['comment_embeddings'][i : i + batch_size].tolist()
        for j, emb in enumerate(embs_batch):
            idx = i + j
            best_cat = ""
            best_sub = ""
            best_kw = ""
            best_score = 0.0
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
            best_scores[idx] = best_score

    feedback_data.drop(columns=['comment_embeddings'], inplace=True)

    chunk_rows = []
    for idx in range(len(feedback_data)):
        row = feedback_data.iloc[idx]
        cat = categories_list[idx]
        subcat = sub_categories_list[idx]
        kwp = keyphrases_list[idx]
        score_ = best_scores[idx]
        if similarity_threshold is not None and score_ < similarity_threshold:
            cat = 'No Match'
            subcat = 'No Match'
        preproc = row['preprocessed_comments']
        sumtext = row['summarized_comments']
        sent = row['sentiment_scores']
        parsed_date = row[date_column].split(' ')[0] if isinstance(row[date_column], str) else None
        hour = pd.to_datetime(row[date_column]).hour if pd.notnull(row[date_column]) else None

        row_ext = row.tolist() + [preproc, sumtext, cat, subcat, kwp, sent, score_, parsed_date, hour]
        chunk_rows.append(row_ext)

    existing_cols = feedback_data.columns.tolist()
    add_cols = [comment_column, 'Summarized Text', 'Category', 'Sub-Category', 'Keyphrase', 'Sentiment', 'Best Match Score', 'Parsed Date', 'Hour']
    headers = existing_cols + add_cols
    out_df = pd.DataFrame(chunk_rows, columns=headers).loc[:, ~pd.DataFrame(chunk_rows, columns=headers).columns.duplicated()]
    return out_df

###################################################################
#   Helper: Summarize top docs into a short label (10 tokens)     #
###################################################################
def summarize_to_short_label(
    docs,
    model_summ,
    tokenizer_summ,
    device,
    # We'll use a 2-step approach:
    # 1) Summarize each doc individually to ~50 tokens
    # 2) Combine them, then summarize to ~10 tokens
    step1_max_length=50,
    step1_min_length=15,
    final_max_length=10,
    final_min_length=3
):
    # Step 1: Summarize each doc individually
    mini_summaries = []
    for d in docs:
        input_ids = tokenizer_summ([d], truncation=True, padding=True, return_tensors='pt')['input_ids'].to(device)
        summary_ids = model_summ.generate(input_ids, max_length=step1_max_length, min_length=step1_min_length)[0]
        stext = tokenizer_summ.decode(summary_ids, skip_special_tokens=True).strip()
        mini_summaries.append(stext)

    # Step 2: Combine mini-summaries
    combined_text = " ".join(mini_summaries)

    # Step 3: Summarize to final short label
    final_ids = tokenizer_summ([combined_text], truncation=True, padding=True, return_tensors='pt')['input_ids'].to(device)
    out_ids = model_summ.generate(final_ids, max_length=final_max_length, min_length=final_min_length)[0]
    final_label = tokenizer_summ.decode(out_ids, skip_special_tokens=True).strip()

    if not final_label:
        final_label = "Emerging Topic"
    return final_label

###################################################################
#   cluster_emerging_issues_bertopic with Keyphrase Prompting     #
###################################################################
def cluster_emerging_issues_bertopic(trends_data, min_topic_size=5, progress=None):
    """
    Single-pass approach with a keyphrase hint in the summarization prompt.

    1. Identify 'No Match' rows.
    2. Fit & transform with BERTopic.
    3. For each topic:
       - build a keyphrase from top words
       - incorporate that into a summarization prompt
       - produce a final subcategory label
    4. Store everything in the DataFrame in one pass.
    """
    # 1) Check if we have any 'No Match' docs
    no_match_mask = (trends_data['Category'] == 'No Match')
    if not no_match_mask.any():
        print("No 'No Match' items found. Skipping BERTopic pass.")
        return trends_data

    # Optionally update the Streamlit progress bar
    if progress:
        progress.progress(5)

    # Subset to just the 'No Match' rows
    df_no_match = trends_data.loc[no_match_mask].copy()
    # If there's a Summarized Text column, we use that; else we use preprocessed
    text_col = 'Summarized Text' if 'Summarized Text' in df_no_match.columns else 'preprocessed_comments'
    no_match_texts = df_no_match[text_col].fillna('').tolist()

    # If fewer than 3 docs, skip clustering
    if len(no_match_texts) < 3:
        print("Not enough 'No Match' docs to cluster. Skipping BERTopic pass.")
        return trends_data

    # If fewer than 15 docs, reduce n_neighbors for UMAP
    n_neighbors = max(2, min(15, len(no_match_texts) - 1))
    print(f"Using n_neighbors={n_neighbors} for {len(no_match_texts)} 'No Match' docs.")

    # 2) Fit a BERTopic model
    topic_model = BERTopic(
        umap_model=UMAP(n_neighbors=n_neighbors, n_components=2, metric="cosine"),
        language="english",
        min_topic_size=min_topic_size,
        embedding_model="all-mpnet-base-v2",
        verbose=True
    )

    if progress:
        progress.progress(10)

    # Actually cluster
    topics, _ = topic_model.fit_transform(no_match_texts)

    if progress:
        progress.progress(60)

    # 3) Summarization for each topic
    model_summ, tokenizer_summ, device = get_summarization_model_and_tokenizer()
    STOPWORDS = set(stopwords.words('english'))

    unique_topics = set(topics)
    for topic_id in unique_topics:
        # Indices of docs in this topic
        idxs = [i for i, t in enumerate(topics) if t == topic_id]

        if topic_id == -1:
            # Mark as truly no match
            df_no_match.iloc[idxs, df_no_match.columns.get_loc('Category')] = 'No Match'
            df_no_match.iloc[idxs, df_no_match.columns.get_loc('Sub-Category')] = 'No Match'
            df_no_match.iloc[idxs, df_no_match.columns.get_loc('Keyphrase')] = 'No Match'
        else:
            # Mark category as Emerging Issues
            df_no_match.iloc[idxs, df_no_match.columns.get_loc('Category')] = 'Emerging Issues'

            # a) Build keyphrase from top words
            top_words = topic_model.get_topic(topic_id)  # list of (word, score)
            filtered = [w for w, _ in top_words if w.lower() not in STOPWORDS and len(w) > 2]
            keyphrase = ", ".join(filtered[:3]) if filtered else "No Keyphrase"
            df_no_match.iloc[idxs, df_no_match.columns.get_loc('Keyphrase')] = keyphrase

            # b) Summarize up to 5 docs to produce Sub-Category
            chosen_idxs = idxs[:5]
            chosen_docs = [no_match_texts[i] for i in chosen_idxs]

            # Prepend a line with the keyphrase
            if keyphrase != "No Keyphrase":
                prompt_line = f"Topic words: {keyphrase}"
                docs_for_label = [prompt_line] + chosen_docs
            else:
                docs_for_label = chosen_docs

            subcat_label = summarize_to_short_label(
                docs_for_label,
                model_summ,
                tokenizer_summ,
                device,
                step1_max_length=50,
                step1_min_length=15,
                final_max_length=25,  # up to 25 tokens
                final_min_length=3
            )

            df_no_match.iloc[idxs, df_no_match.columns.get_loc('Sub-Category')] = subcat_label

    # 4) Update the original dataframe with our emergent categories
    trends_data.update(df_no_match)

    if progress:
        progress.progress(100)

    return trends_data


###############################################
#          MAIN STREAMLIT APPLICATION         #
###############################################
st.set_page_config(layout="wide")
st.title("\U0001F9D1\u200D\U0001F4BB Transcript Categorization")

model = initialize_bert_model()

emerging_issue_mode = st.sidebar.checkbox("Emerging Issue Mode")
similarity_threshold = None
if emerging_issue_mode:
    similarity_threshold = st.sidebar.slider("Semantic Similarity Threshold", 0.0, 1.0, 0.35)
    min_topic_size = st.sidebar.slider("Minimum Topic Size", 3, 20, 5)

st.sidebar.header("Edit Categories")
ui_new_categories = {}
for category, subcategories in default_categories.items():
    category_name = st.sidebar.text_input(f"{category} Category", value=category)
    new_subs = {}
    for subcategory, kwds in subcategories.items():
        subcat_name = st.sidebar.text_input(f"{subcategory} Subcategory under {category_name}", value=subcategory)
        with st.sidebar.expander(f"Keywords for {subcat_name}"):
            category_keywords = st.text_area("Keywords", value="\n".join(kwds))
        new_subs[subcat_name] = category_keywords.split("\n")
    ui_new_categories[category_name] = new_subs
default_categories = ui_new_categories

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
        st.stop()

    comment_column = st.selectbox("Select the column containing the comments", column_names)
    date_column = st.selectbox("Select the column containing the dates", column_names)
    grouping_option = st.radio("Select how to group the dates", ["Date", "Week", "Month", "Quarter", "Hour"])
    process_button = st.button("Process Feedback")

    progress_bar = st.progress(0)
    processed_chunks = []
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

        for i, feedback_data in enumerate(chunk_iter):
            chunk_result = process_feedback_data_chunk(
                feedback_data,
                comment_column,
                date_column,
                default_categories,
                similarity_threshold
            )
            processed_chunks.append(chunk_result)

            partial_data = pd.concat(processed_chunks, ignore_index=True)

            if not partial_data.empty:
                trends_dataframe_placeholder.dataframe(partial_data)

                partial_data['Parsed Date'] = pd.to_datetime(partial_data['Parsed Date'], errors='coerce')
                if grouping_option == 'Date':
                    pivot = partial_data.pivot_table(
                        index=['Category', 'Sub-Category'],
                        columns=pd.Grouper(key='Parsed Date', freq='D'),
                        values='Sentiment',
                        aggfunc='count',
                        fill_value=0
                    )
                elif grouping_option == 'Week':
                    pivot = partial_data.pivot_table(
                        index=['Category', 'Sub-Category'],
                        columns=pd.Grouper(key='Parsed Date', freq='W-SUN', closed='left', label='left'),
                        values='Sentiment',
                        aggfunc='count',
                        fill_value=0
                    )
                elif grouping_option == 'Month':
                    pivot = partial_data.pivot_table(
                        index=['Category', 'Sub-Category'],
                        columns=pd.Grouper(key='Parsed Date', freq='M'),
                        values='Sentiment',
                        aggfunc='count',
                        fill_value=0
                    )
                elif grouping_option == 'Quarter':
                    pivot = partial_data.pivot_table(
                        index=['Category', 'Sub-Category'],
                        columns=pd.Grouper(key='Parsed Date', freq='Q'),
                        values='Sentiment',
                        aggfunc='count',
                        fill_value=0
                    )
                elif grouping_option == 'Hour':
                    if 'Hour' not in partial_data.columns:
                        partial_data['Hour'] = pd.to_datetime(partial_data[date_column]).dt.hour
                    pivot = partial_data.pivot_table(
                        index=['Category', 'Sub-Category'],
                        columns='Hour',
                        values='Sentiment',
                        aggfunc='count',
                        fill_value=0
                    )

                pivot.columns = pivot.columns.astype(str)
                pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
                pivot = pivot[sorted(pivot.columns, reverse=True)]

                pivot_reset = pivot.reset_index().set_index('Sub-Category').drop(columns=['Category'], errors='ignore')
                top_5_trends = pivot_reset.head(5).T
                line_chart_placeholder.line_chart(top_5_trends)
                pivot_table_placeholder.dataframe(pivot)

                pivot2 = partial_data.groupby(['Category', 'Sub-Category'])['Sentiment'].agg(['mean', 'count'])
                pivot2.columns = ['Average Sentiment', 'Quantity']
                pivot2 = pivot2.sort_values('Quantity', ascending=False)
                pivot2_reset = pivot2.reset_index().set_index('Sub-Category')

                subcategory_sentiment_dataframe_placeholder.dataframe(pivot2_reset)
                subcategory_sentiment_bar_chart_placeholder.bar_chart(pivot2_reset['Quantity'])

                top_subcategories = pivot2_reset.head(10).index.tolist()
                for idx, subcat in enumerate(top_subcategories[:10]):
                    title_placeholder, table_placeholder = combined_placeholders[idx]
                    title_placeholder.subheader(f"[CHUNK {i+1}] {subcat}")
                    filtered_data = partial_data[partial_data['Sub-Category'] == subcat].copy()
                    filtered_data['Parsed Date'] = pd.to_datetime(filtered_data['Parsed Date'], errors='coerce')
                    top_comments = (
                        filtered_data
                        .dropna(subset=["Parsed Date"])
                        .sort_values("Parsed Date", ascending=False)
                        .head(10)
                        [[
                            'Parsed Date',
                            comment_column,
                            'Summarized Text',
                            'Keyphrase',
                            'Sentiment',
                            'Best Match Score'
                        ]]
                    )
                    top_comments['Parsed Date'] = top_comments['Parsed Date'].dt.date.astype(str)
                    table_placeholder.table(top_comments)

            processed_chunks_count += 1
            progress_bar.progress(processed_chunks_count / estimated_total_chunks)

        trends_data = pd.concat(processed_chunks, ignore_index=True)

        # The final pass with a mini progress bar:
        if emerging_issue_mode:
            st.write("Performing BERTopic clustering on No Match items...")
            final_pass_bar = st.progress(0)
            # pass final_pass_bar into cluster_emerging_issues_bertopic
            trends_data = cluster_emerging_issues_bertopic(
                trends_data,
                min_topic_size=min_topic_size,
                progress=final_pass_bar
            )
            final_pass_bar.progress(100)

        if not trends_data.empty:
            trends_dataframe_placeholder.dataframe(trends_data)

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
                    trends_data['Hour'] = pd.to_datetime(trends_data[date_column]).dt.hour
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

            pivot_reset = pivot.reset_index().set_index('Sub-Category').drop(columns=['Category'], errors='ignore')
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
                title_placeholder.subheader(f"FINAL {subcat}")
                filtered_data = trends_data[trends_data['Sub-Category'] == subcat].copy()
                filtered_data['Parsed Date'] = pd.to_datetime(filtered_data['Parsed Date'], errors='coerce')

                top_comments = (
                    filtered_data
                    .dropna(subset=["Parsed Date"])
                    .sort_values("Parsed Date", ascending=False)
                    .head(10)
                    [[
                        "Parsed Date", 
                        comment_column, 
                        "Summarized Text", 
                        "Keyphrase", 
                        "Sentiment", 
                        "Best Match Score"
                    ]]
                )
                top_comments['Parsed Date'] = top_comments['Parsed Date'].dt.date.astype(str)
                table_placeholder.table(top_comments)

            # Convert dates to strings for final pivot
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
        with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
            trends_data.to_excel(writer, sheet_name='Feedback Trends', index=False)
            pivot.to_excel(writer, sheet_name=f'Trends by {grouping_option}', merge_cells=False)
            pivot1.to_excel(writer, sheet_name='Categories', merge_cells=False)
            pivot2.to_excel(writer, sheet_name='Subcategories', merge_cells=False)
            example_comments_sheet = writer.book.add_worksheet('Example Comments')

            # Unique enumerated merges
            for idx, subcat in enumerate(top_subcategories):
                filtered_data = trends_data[trends_data['Sub-Category'] == subcat]
                filtered_data['Parsed Date'] = pd.to_datetime(filtered_data['Parsed Date'], errors='coerce')
                ex_top_comments = (
                    filtered_data
                    .dropna(subset=["Parsed Date"])
                    .sort_values("Parsed Date", ascending=False)
                    .head(10)
                    [["Parsed Date", comment_column]]
                )
                start_row = (idx * 8) + 1
                example_comments_sheet.merge_range(start_row, 0, start_row, 1, subcat)
                example_comments_sheet.write(start_row + 1, 0, 'Date')
                example_comments_sheet.write(start_row + 1, 1, comment_column)
                for rowi, (_, rowv) in enumerate(ex_top_comments.iterrows(), start=start_row + 2):
                    example_comments_sheet.write(rowi, 0, str(rowv['Parsed Date']))
                    example_comments_sheet.write_string(rowi, 1, str(rowv[comment_column]))

        excel_file.seek(0)
        b64 = base64.b64encode(excel_file.read()).decode()
        download_link_placeholder.markdown(
            f'<a href="data:application/octet-stream;base64,{b64}" download="feedback_trends.xlsx">Download Excel File</a>',
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    st.write("Transcript Categorization App")
