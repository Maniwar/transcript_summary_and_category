import os

# Set the environment variable to control tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # or "false" to enable or disable parallelism

# Now you can import the rest of the modules
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import numpy as np
import xlsxwriter
import chardet
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import base64
from io import BytesIO
import streamlit as st
import textwrap
from categories_josh_sub_V5 import default_categories
import time
from tqdm import tqdm
import re
import string
import unicodedata
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
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
        tokens = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze()


# Initialize BERT model
@st.cache_resource

def initialize_bert_model():
    start_time = time.time()
    print("Initializing BERT model...")
    end_time = time.time()
    print(f"BERT model initialized. Time taken: {end_time - start_time} seconds.")
    return SentenceTransformer('paraphrase-MiniLM-L12-v2', device="cpu")

# Initialize a variable to store the previous state of the categories
previous_categories = None

# Function to compute keyword embeddings
def compute_keyword_embeddings(categories):
    start_time = time.time()
    print("Computing keyword embeddings...")
    model = initialize_bert_model()
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
@st.cache_data
def preprocess_text(text):
    #start_time = time.time()
    #print("Preprocessing text...")
    # Convert to string if input is a float
    if isinstance(text, float):
        text = str(text)
    #end_time = time.time()
    #print(f"Preprocessing text completed. Time taken: {end_time - start_time} seconds.")
    # Remove emojis and special characters
    text = text.encode('ascii', 'ignore').decode('utf-8')
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
@st.cache_data
def perform_sentiment_analysis(text):
    #start_time = time.time()
    #print("Perform Sentiment Analysis text...")
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    #end_time = time.time()
    #print(f"Sentiment Analysis completed. Time taken: {end_time - start_time} seconds.")
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
            # If a single comment exceeds max_tokens, split it and add it to the chunks
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
            # If adding the current comment exceeds max_tokens, finalize the current chunk
            if current_chunk_tokens + tokens > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [comment]
                current_chunk_tokens = tokens
            else:
                current_chunk.append(comment)
                current_chunk_tokens += tokens

    # Add any remaining comments to the chunks
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Print the chunking results
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

def preprocess_comments_and_summarize(feedback_data, comment_column, batch_size=16, max_length=75, min_length=30, max_tokens=1000, very_short_limit=30):
    print("Starting preprocessing and summarization...")

    # 1. Preprocess the comments
    feedback_data['preprocessed_comments'] = feedback_data[comment_column].apply(preprocess_text)
    print("Comments preprocessed.")

    # 2. Get the cached model and tokenizer
    model, tokenizer, device = get_summarization_model_and_tokenizer()
    print("Summarization model and tokenizer retrieved from cache.")

    # 3. Separate comments into categories
    very_short_comments = [comment for comment in feedback_data['preprocessed_comments'].tolist() if get_token_count(comment, tokenizer) <= very_short_limit]
    short_comments = [comment for comment in feedback_data['preprocessed_comments'].tolist() if very_short_limit < get_token_count(comment, tokenizer) <= max_tokens]
    long_comments = [comment for comment in feedback_data['preprocessed_comments'].tolist() if get_token_count(comment, tokenizer) > max_tokens]
    print(f"Separated comments into categories: {len(very_short_comments)} very short, {len(short_comments)} short, and {len(long_comments)} long comments.")

    # 4. Handle very short comments
    summaries_dict = {comment: comment for comment in very_short_comments}
    print(f"{len(very_short_comments)} very short comments directly added to summaries.")

    # 5. Handle short comments
    pbar = tqdm(total=len(short_comments), desc="Summarizing short comments")
    for i in range(0, len(short_comments), batch_size):
        batch = short_comments[i:i+batch_size]
        summaries = [summarize_text(comment, tokenizer, model, device, max_length, min_length) for comment in batch]
        for original_comment, summary in zip(batch, summaries):
            summaries_dict[original_comment] = summary
        pbar.update(len(batch))
    pbar.close()

    # 6. Handle long comments
    pbar = tqdm(total=len(long_comments), desc="Summarizing long comments")
    for comment in long_comments:
        chunks = split_comments_into_chunks([(comment, get_token_count(comment, tokenizer))], tokenizer, max_tokens)
        summaries = [summarize_text(chunk, tokenizer, model, device, max_length, min_length) for chunk in chunks]
        full_summary = " ".join(summaries)

        resummarization_count = 0
        while get_token_count(full_summary, tokenizer) > max_length:
            resummarization_count += 1
            print(f"Re-summarizing a long comment with token count: {get_token_count(full_summary, tokenizer)}")
            full_summary = summarize_text(full_summary, tokenizer, model, device, max_length, min_length)

        # Display the number of times a comment was re-summarized
        if resummarization_count > 0:
            print(f"Long comment was re-summarized {resummarization_count} times to fit the max length.")

        summaries_dict[comment] = full_summary
        pbar.update(1)
    pbar.close()

    print("Preprocessing and summarization completed.")
    return summaries_dict


def summarize_text(text, tokenizer, model, device, max_length, min_length):
    input_ids = tokenizer([text], truncation=True, padding=True, return_tensors='pt')['input_ids'].to(device)
    summary_ids = model.generate(input_ids, max_length=max_length, min_length=min_length)[0]
    return tokenizer.decode(summary_ids, skip_special_tokens=True)


# Function to compute semantic similarity
def compute_semantic_similarity(comment_embedding, keyword_embedding):
    return cosine_similarity([comment_embedding], [keyword_embedding])[0][0]
# Set the default layout mode to "wide"
st.set_page_config(layout="wide")

# Streamlit interface
st.title("👨‍💻 Transcript Categorization")

# Add checkbox for emerging issue mode
emerging_issue_mode = st.sidebar.checkbox("Emerging Issue Mode")

# Sidebar description for emerging issue mode
st.sidebar.write("Emerging issue mode allows you to set a minimum similarity score. If the comment doesn't match up to the categories based on the threshold, it will be set to NO MATCH.")

# Add slider for semantic similarity threshold in emerging issue mode
similarity_threshold = None
similarity_score = None
best_match_score = None

if emerging_issue_mode:
    similarity_threshold = st.sidebar.slider("Semantic Similarity Threshold", min_value=0.0, max_value=1.0, value=0.35)

# Initialize an empty dictionary for categories
categories = {}

# Edit categories, subcategories and keywords
# Edit categories and keywords
st.sidebar.header("Edit Categories")

# Create a new dictionary to store the updated categories
new_categories = {}

# Iterate over each category and its subcategories
for category, subcategories in default_categories.items():
    # Create a text input field for the category
    category_name = st.sidebar.text_input(f"{category} Category", value=category)

    # Create a new dictionary to store the updated subcategories
    new_subcategories = {}

    # Iterate over each subcategory and its keywords
    for subcategory, keywords in subcategories.items():
        # Create a text input field for the subcategory
        subcategory_name = st.sidebar.text_input(f"{subcategory} Subcategory under {category_name}", value=subcategory)

        # Create a text area for the keywords
        with st.sidebar.expander(f"Keywords for {subcategory_name}"):
            category_keywords = st.text_area("Keywords", value="\n".join(keywords))

        # Update the keywords in the new_subcategories dictionary
        new_subcategories[subcategory_name] = category_keywords.split("\n")

    # Update the subcategories in the new_categories dictionary
    new_categories[category_name] = new_subcategories

# Replace the original default_categories dictionary with the new_categories dictionary
default_categories = new_categories

# Text field for user input
user_input = st.text_area("Enter your transcript")

# Process button
process_button = st.button("Process Transcript")

if user_input and process_button:
    # Process the user input here
    feedback_data = pd.DataFrame([user_input], columns=['comment'])



    @st.cache_data
    def process_feedback_data(feedback_data, comment_column, categories, similarity_threshold):
        global previous_categories
        if previous_categories != categories:  # Use the categories parameter here
            # Compute keyword embeddings
            keyword_embeddings = compute_keyword_embeddings(categories)  # And here
            # Update the previous state of the categories
            previous_categories = categories.copy()  # And here


        # Initialize lists for categorized_comments, sentiments, similarity scores, and summaries
        categorized_comments = []
        sentiments = []
        similarity_scores = []
        summarized_texts = []
        categories_list = []

        # Initialize the BERT model once
        model = initialize_bert_model()


        # Preprocess comments and summarize if necessary
        start_time = time.time()
        print("Preprocessing comments and summarizing if necessary...")

        summaries_dict = preprocess_comments_and_summarize(feedback_data, comment_column)

        # Create a new column for the summarized comments
        feedback_data['preprocessed_comments'] = feedback_data[comment_column].apply(preprocess_text)
        feedback_data['summarized_comments'] = feedback_data['preprocessed_comments'].map(summaries_dict)

        # Fill in missing summarized comments with the original preprocessed comments
        feedback_data['summarized_comments'] = feedback_data['summarized_comments'].fillna(feedback_data['preprocessed_comments'])

        end_time = time.time()
        print(f"Preprocessed comments and summarized. Time taken: {end_time - start_time} seconds.")


        # Compute comment embeddings in batches
        start_time = time.time()
        print("Start comment embeddings in batches")
        batch_size = 1024  # Choose batch size based on your available memory
        comment_embeddings = []
        for i in range(0, len(feedback_data), batch_size):
            batch = feedback_data['summarized_comments'][i:i+batch_size].tolist()
            comment_embeddings.extend(model.encode(batch))
        feedback_data['comment_embeddings'] = comment_embeddings
        end_time = time.time()
        print(f"Batch comment embeddings done. Time taken: {end_time - start_time} seconds.")

        # Compute sentiment scores
        start_time = time.time()
        print("Computing sentiment scores...")
        feedback_data['sentiment_scores'] = feedback_data['preprocessed_comments'].apply(perform_sentiment_analysis)
        end_time = time.time()
        print(f"Sentiment scores computed. Time taken: {end_time - start_time} seconds.")

        # Compute semantic similarity and assign categories in batches
        start_time = time.time()
        print("Computing semantic similarity and assigning categories...")
        # Initialize categories_list, sub_categories_list, keyphrases_list, summarized_texts, and similarity_scores with empty strings and zeros
        categories_list = [''] * len(feedback_data)
        sub_categories_list = [''] * len(feedback_data)
        keyphrases_list = [''] * len(feedback_data)
        summarized_texts = [''] * len(feedback_data)
        similarity_scores = [0.0] * len(feedback_data)

        # Initialize a dictionary to store the similarity scores for all keyphrases
        similarity_scores_all = {keyphrase: [0.0] * len(feedback_data) for _, _, keyphrase in keyword_embeddings.keys()}

        for i in range(0, len(feedback_data), batch_size):
            batch_embeddings = feedback_data['comment_embeddings'][i:i + batch_size].tolist()
            for (category, subcategory, keyword), embeddings in keyword_embeddings.items():
                batch_similarity_scores = [compute_semantic_similarity(batch_embedding, embeddings) for batch_embedding in batch_embeddings]
                # Update categories, sub-categories, and keyphrases based on the highest similarity score
                for j, similarity_score in enumerate(batch_similarity_scores):
                    idx = i + j  # Index in the complete list
                    if idx < len(categories_list):
                        if similarity_score > similarity_scores[idx]:
                            categories_list[idx] = category
                            sub_categories_list[idx] = subcategory
                            keyphrases_list[idx] = keyword
                            summarized_texts[idx] = keyword
                            similarity_scores[idx] = similarity_score
                    else:
                        categories_list.append(category)
                        sub_categories_list.append(subcategory)
                        keyphrases_list.append(keyword)
                        summarized_texts.append(keyword)
                        similarity_scores.append(similarity_score)
                    # Store the similarity score for the current keyphrase
                    similarity_scores_all[keyword][idx] = similarity_score

        end_time = time.time()
        print(f"Computed semantic similarity and assigned categories. Time taken: {end_time - start_time} seconds.")

        # Prepare final data
        for index, row in feedback_data.iterrows():
            preprocessed_comment = row['preprocessed_comments']
            sentiment_score = row['sentiment_scores']
            category = categories_list[index]
            sub_category = sub_categories_list[index]
            keyphrase = keyphrases_list[index]
            best_match_score = similarity_scores[index]
            summarized_text = row['summarized_comments']

            # If in emerging issue mode and the best match score is below the threshold, set category, sub-category, and keyphrase to 'No Match'
            if emerging_issue_mode and best_match_score < similarity_threshold:
                category = 'No Match'
                sub_category = 'No Match'
                keyphrase = 'No Match'


            row_extended = row.tolist() + [preprocessed_comment, summarized_text, category, sub_category, keyphrase, sentiment_score, best_match_score]
            categorized_comments.append(row_extended)

        # Create a new DataFrame with extended columns
        existing_columns = feedback_data.columns.tolist()
        additional_columns = [comment_column, 'Summarized Text', 'Category', 'Sub-Category', 'Keyphrase', 'Sentiment', 'Best Match Score']
        headers = existing_columns + additional_columns
        trends_data = pd.DataFrame(categorized_comments, columns=headers)

        # Rename duplicate column names
        trends_data = trends_data.loc[:, ~trends_data.columns.duplicated()]
        duplicate_columns = set([col for col in trends_data.columns if trends_data.columns.tolist().count(col) > 1])
        for column in duplicate_columns:
            column_indices = [i for i, col in enumerate(trends_data.columns) if col == column]
            for i, idx in enumerate(column_indices[1:], start=1):
                trends_data.columns.values[idx] = f"{column}_{i}"

        return trends_data, best_match_score, similarity_scores, similarity_scores_all


    # Call the function with the correct dictionary format
    trends_data, best_match_score, similarity_scores, similarity_scores_all = process_feedback_data(feedback_data, 'comment', default_categories, similarity_threshold)

    # Display the summarized text
    st.subheader("Summarized Text")
    if not trends_data.empty:
        st.write(trends_data['Summarized Text'].values[0])

    # Show the best match and its score along with related category, subcategory, and keyphrase
    st.subheader("Best Match and Related Information")
    if not trends_data.empty:
        best_match_category = trends_data['Category'].values[0]
        best_match_subcategory = trends_data['Sub-Category'].values[0]
        best_match_keyphrase = trends_data['Keyphrase'].values[0]

        # Use Streamlit's st.columns to display information side by side
        col1, col2 = st.columns(2)

        # Display the information in columns
        with col1:
            st.write("Category:", best_match_category)
            st.write("Subcategory:", best_match_subcategory)
            st.write("Keyphrase:", best_match_keyphrase)
        with col2:
            st.write("Score:", best_match_score)

    # Convert the dictionary to a DataFrame
    df_similarity_scores_all = pd.DataFrame(list(similarity_scores_all.items()), columns=['Keyphrase', 'Similarity Score'])

    # Sort the DataFrame by 'Similarity Score' in descending order
    df_similarity_scores_all = df_similarity_scores_all.sort_values('Similarity Score', ascending=False)

    # Display the sorted DataFrame along with related category and subcategory
    st.subheader("Similarity Scores for All Keyphrases")
    if not df_similarity_scores_all.empty:
        st.dataframe(df_similarity_scores_all)

    # Add additional space to visually separate sections
    st.write("")
