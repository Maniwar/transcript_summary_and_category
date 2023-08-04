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
from categories_josh_sub import default_categories
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
    return SentenceTransformer('paraphrase-MiniLM-L12-v2', device="cuda")

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
    start_time = time.time()
    print("Preprocessing text...")
    # Convert to string if input is a float
    if isinstance(text, float):
        text = str(text)
    end_time = time.time()
    print(f"Preprocessing text completed. Time taken: {end_time - start_time} seconds.")
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
    start_time = time.time()
    print("Perform Sentiment Analysis text...")
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    end_time = time.time()
    print(f"Sentiment Analysis completed. Time taken: {end_time - start_time} seconds.")
    return compound_score

# Function to compute the token count of a text
def get_token_count(text, tokenizer):
    return len(tokenizer.encode(text)) - 2

# Function to split comments into chunks
def split_comments_into_chunks(comments, tokenizer, max_tokens_per_chunk):
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0

    for comment, tokens in comments:
        if tokens > max_tokens_per_chunk:
            parts = textwrap.wrap(comment, width=max_tokens_per_chunk)
            for part in parts:
                part_tokens = get_token_count(part, tokenizer)
                if current_chunk_tokens + part_tokens > max_tokens_per_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [part]
                    current_chunk_tokens = part_tokens
                else:
                    current_chunk.append(part)
                    current_chunk_tokens += part_tokens
        else:
            if current_chunk_tokens + tokens > max_tokens_per_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [comment]
                current_chunk_tokens = tokens
            else:
                current_chunk.append(comment)
                current_chunk_tokens += tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))
     # Print the chunking results
    print(f"Total number of chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} token count: {get_token_count(chunk, tokenizer)}")
    return chunks

def preprocess_comments_and_summarize(feedback_data, comment_column, batch_size=16, max_length=100, min_length=30, max_tokens=1000, min_word_count=100):
    print("Preprocessing comments and summarizing if necessary...")

    # Preprocess the comments
    feedback_data['preprocessed_comments'] = feedback_data[comment_column].apply(preprocess_text)

    # Initialize the summarization model and tokenizer
    model_name = "knkarthick/MEETING_SUMMARY"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)  # Move the model to the selected device

    # Separate comments into short and long based on token count
    short_comments = [comment for comment in feedback_data['preprocessed_comments'].tolist() if get_token_count(comment, tokenizer) <= min_word_count]
    long_comments = [comment for comment in feedback_data['preprocessed_comments'].tolist() if get_token_count(comment, tokenizer) > min_word_count]

    # Initialize a dictionary to store the summaries
    summaries_dict = {}

    # Process short comments in batches
    for i in range(0, len(short_comments), batch_size):
        batch = short_comments[i:i+batch_size]
        input_ids = tokenizer(batch, truncation=True, padding=True, return_tensors='pt')['input_ids'].to(device)
        summaries = model.generate(input_ids, max_length=max_length, min_length=min_length)
        for original_text, summary_ids in zip(batch, summaries):
            summary_text = tokenizer.decode(summary_ids, skip_special_tokens=True)
            summaries_dict[original_text] = summary_text

    # Create a list of chunks from long comments
    chunks_with_original = []
    for comment in long_comments:
        chunks = split_comments_into_chunks([(comment, get_token_count(comment, tokenizer))], tokenizer, max_tokens)
        for chunk in chunks:
            chunks_with_original.append((comment, chunk))

    # Process chunks in batches
    chunk_summaries = defaultdict(list)
    for i in range(0, len(chunks_with_original), batch_size):
        batch_with_original = chunks_with_original[i:i+batch_size]
        batch = [chunk for _, chunk in batch_with_original]
        input_ids = tokenizer(batch, truncation=True, padding=True, return_tensors='pt')['input_ids'].to(device)
        summaries = model.generate(input_ids, max_length=max_length, min_length=min_length)
        for (original_text, _), summary_ids in zip(batch_with_original, summaries):
            summary_text = tokenizer.decode(summary_ids, skip_special_tokens=True)
            chunk_summaries[original_text].append(summary_text)

    # Stitch together and re-summarize if needed
    for original_text, summary_parts in chunk_summaries.items():
        full_summary = " ".join(summary_parts)
        print(f"Stitching {len(summary_parts)} summary parts for original text with token count: {get_token_count(original_text, tokenizer)}")
        while get_token_count(full_summary, tokenizer) > max_length:
            print(f"Re-summarizing text with token count: {get_token_count(full_summary, tokenizer)}")
            # Re-summarize the full summary if needed
            input_ids = tokenizer([full_summary], truncation=True, padding=True, return_tensors='pt')['input_ids'].to(device)
            summary_ids = model.generate(input_ids, max_length=max_length, min_length=min_length)[0]
            full_summary = tokenizer.decode(summary_ids, skip_special_tokens=True)
        print(f"Final summary token count: {get_token_count(full_summary, tokenizer)}")
        summaries_dict[original_text] = full_summary



    return summaries_dict



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
