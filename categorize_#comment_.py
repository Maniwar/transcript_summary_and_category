import os

# Set the environment variable to control tokenizers parallelism
# This line sets an environment variable that can influence how tokenizers (used by transformer models) handle parallelism.
# "TOKENIZERS_PARALLELISM" can be set to "true" to enable parallel processing if available, potentially speeding up tokenization,
# or "false" to disable it, which might be beneficial in certain environments or to reduce overhead.
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # or "false" to enable or disable parallelism

# Now you can import the rest of the modules
# Import necessary libraries for data manipulation, NLP, machine learning, and Streamlit app development.
import torch # PyTorch library for tensor computations and neural networks.
from torch.utils.data import Dataset # For creating custom datasets for PyTorch models.
from torch.utils.data import DataLoader # For loading data in batches for PyTorch models.
import pandas as pd # Pandas for data manipulation and analysis using DataFrames.
import nltk # Natural Language Toolkit for text processing tasks.
from nltk.sentiment import SentimentIntensityAnalyzer # NLTK's sentiment analyzer for determining sentiment polarity of text.
from sentence_transformers import SentenceTransformer # Sentence Transformers library for creating sentence embeddings.
from sklearn.metrics.pairwise import cosine_similarity # scikit-learn for calculating cosine similarity between vectors.
import datetime # Python's datetime module for working with dates and times.
import numpy as np # NumPy for numerical operations and array manipulations.
import xlsxwriter # For writing data to Excel files.
import chardet # For detecting the character encoding of a file.
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM # Hugging Face Transformers library for NLP models and tokenizers.
import base64 # For encoding binary data to ASCII strings and vice versa.
from io import BytesIO # For working with in-memory binary streams.
import streamlit as st # Streamlit for creating interactive web applications.
import textwrap # For wrapping and filling text.
from categories_josh_sub_V6_3 import default_categories # Importing predefined categories and keywords from a separate file.
import time # Python's time module for time-related functions, like measuring execution time.
from tqdm import tqdm # For displaying progress bars, useful for long-running loops.
import re # Regular expression operations for text pattern matching.
import string # String constants and classes (e.g., string.punctuation).
import unicodedata # Unicode database access.
import math # Math functions.

from collections import defaultdict # Dictionary-like class that calls a factory function to supply missing values.

# Define a custom PyTorch Dataset for text summarization.
class SummarizationDataset(Dataset):
    # Initialize the dataset with texts, tokenizer, and maximum sequence length.
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts # List of input texts to be summarized.
        self.tokenizer = tokenizer # Tokenizer object from Hugging Face Transformers to convert text to tokens.
        self.max_length = max_length # Maximum length for tokenized sequences.

    # Return the number of items in the dataset (number of texts).
    def __len__(self):
        return len(self.texts)

    # Get an item from the dataset at a given index.
    def __getitem__(self, idx):
        text = self.texts[idx] # Get the text at the specified index.
        # Tokenize the text:
        # - truncation=True:  Truncate sequences longer than max_length.
        # - padding='max_length': Pad sequences shorter than max_length to max_length.
        # - max_length=self.max_length: Set the maximum length for tokenization.
        # - return_tensors='pt': Return PyTorch tensors.
        tokens = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        # Return input IDs (token indices) and attention mask (to ignore padded tokens).
        return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze() # .squeeze() removes dimensions of size 1.


# Initialize BERT model for sentence embeddings.
# @st.cache_resource is a Streamlit decorator that caches the output of this function.
# It's used for expensive operations like model loading, so it's only run once and reused across sessions.
@st.cache_resource
def initialize_bert_model():
    start_time = time.time() # Record start time to measure initialization duration.
    print("Initializing BERT model...") # Print message to console.
    end_time = time.time() # Record end time.
    print(f"BERT model initialized. Time taken: {end_time - start_time} seconds.") # Print initialization time.
    # Load a pre-trained SentenceTransformer model. 'all-mpnet-base-v2' is a good general-purpose model.
    # device="cpu" forces the model to run on CPU. Change to "cuda" if you have a GPU and want to use it.
    return SentenceTransformer('all-mpnet-base-v2', device="cpu")

# Initialize a variable to store the previous state of the categories.
# This is used for caching to avoid recomputing keyword embeddings unnecessarily.
previous_categories = None

# Function to compute embeddings for keywords in the categories.
# @st.cache_data(persist="disk") is a Streamlit decorator that caches the output of this function.
# persist="disk" means the cache will be saved to disk and reused across app runs, as long as the input 'categories' is the same.
@st.cache_data(persist="disk")
def compute_keyword_embeddings(categories):
    start_time = time.time() # Record start time.
    print("Computing keyword embeddings...") # Print message to console.
    #model = initialize_bert_model() # This line is commented out, assuming 'model' is initialized globally outside.
    keyword_embeddings = {} # Initialize an empty dictionary to store keyword embeddings.

    # Iterate through each category and its subcategories and keywords.
    for category, subcategories in categories.items():
        for subcategory, keywords in subcategories.items():
            for keyword in keywords:
                # Check if embedding for this (category, subcategory, keyword) combination is already computed.
                if (category, subcategory, keyword) not in keyword_embeddings:
                    # If not, compute the embedding for the keyword using the BERT model.
                    # model.encode([keyword]) encodes the keyword into a sentence embedding. [0] gets the embedding vector.
                    keyword_embeddings[(category, subcategory, keyword)] = model.encode([keyword])[0]

    end_time = time.time() # Record end time.
    print(f"Keyword embeddings computed. Time taken: {end_time - start_time} seconds.") # Print computation time.
    return keyword_embeddings # Return the dictionary of keyword embeddings.

# Function to preprocess the input text to clean and standardize it.
def preprocess_text(text):
    #start_time = time.time()
    #print("Preprocessing text...")
    # Convert to string if input is a float. Sometimes data from CSV can be read as float if it's numeric.
    if isinstance(text, float):
        text = str(text)
    #end_time = time.time()
    #print(f"Preprocessing text completed. Time taken: {end_time - start_time} seconds.")
    # Remove emojis and special characters by encoding to ASCII and ignoring errors, then decoding back.
    # This is a basic way to remove non-ASCII characters. 'encoding' variable should be defined elsewhere (likely when reading CSV).
    text = text.encode('ascii', 'ignore').decode(encoding)
    # Remove punctuation using regular expressions. r'[^\w\s]' matches any character that is NOT a word character (\w) or whitespace (\s).
    text = re.sub(r'[^\w\s]', '', text)
    # Remove HTML tags using regular expressions.  r'<.*?>' matches anything between '<' and '>'.
    text = re.sub(r'<.*?>', '', text)
    # Remove page breaks (\n and \r) and replace them with spaces.
    text = text.replace('\n', ' ').replace('\r', '')
    # Remove non-breaking spaces ( ) and replace them with regular spaces.
    text = text.replace(' ', ' ')
    # Remove multiple spaces and trim leading/trailing spaces. r'\s+' matches one or more whitespace characters.
    text = re.sub(r'\s+', ' ', text).strip()
    return text # Return the preprocessed text.


# Function to perform sentiment analysis on a given text.
def perform_sentiment_analysis(text):
    #start_time = time.time()
    #print("Perform Sentiment Analysis text...")
    analyzer = SentimentIntensityAnalyzer() # Initialize NLTK's SentimentIntensityAnalyzer.
    sentiment_scores = analyzer.polarity_scores(text) # Get sentiment scores (negative, neutral, positive, compound).
    compound_score = sentiment_scores['compound'] # Extract the compound sentiment score, which is a single metric from -1 (most extreme negative) to +1 (most extreme positive).
    #end_time = time.time()
    #print(f"Sentiment Analysis completed. Time taken: {end_time - start_time} seconds.")
    return compound_score # Return the compound sentiment score.


# Function to compute the number of tokens in a text using a tokenizer.
def get_token_count(text, tokenizer):
    # tokenizer.encode(text) tokenizes the text and adds special tokens like [CLS] and [SEP].
    # We subtract 2 to exclude the count of [CLS] and [SEP] tokens, as we are interested in the actual content tokens.
    return len(tokenizer.encode(text)) - 2

# Function to split comments into chunks so that each chunk does not exceed a maximum number of tokens.
def split_comments_into_chunks(comments, tokenizer, max_tokens):
    # Sort comments in descending order based on their token count.
    # This is done to process longer comments first, potentially filling up chunks more efficiently.
    sorted_comments = sorted(comments, key=lambda x: x[1], reverse=True)

    chunks = [] # Initialize an empty list to store the chunks of comments.
    current_chunk = [] # Initialize an empty list to build the current chunk.
    current_chunk_tokens = 0 # Initialize token count for the current chunk.

    # Iterate through the sorted comments (each comment is a tuple of (text, token_count)).
    for comment, tokens in sorted_comments:
        if tokens > max_tokens:
            # If a single comment's token count is already greater than max_tokens, split it into smaller parts.
            # textwrap.wrap breaks the comment into lines of width max_tokens.
            parts = textwrap.wrap(comment, width=max_tokens)
            for part in parts: # Iterate through each part of the split comment.
                part_tokens = get_token_count(part, tokenizer) # Get token count of the part.
                # Check if adding this part to the current chunk exceeds max_tokens.
                if current_chunk_tokens + part_tokens > max_tokens:
                    chunks.append(" ".join(current_chunk)) # If it exceeds, finalize the current chunk by joining the comments and add to chunks list.
                    current_chunk = [part] # Start a new chunk with the current part.
                    current_chunk_tokens = part_tokens # Update token count for the new chunk.
                else:
                    current_chunk.append(part) # If it doesn't exceed, add the part to the current chunk.
                    current_chunk_tokens += part_tokens # Update token count for the current chunk.
        else:
            # If the comment's token count is within max_tokens.
            # Check if adding this comment to the current chunk exceeds max_tokens.
            if current_chunk_tokens + tokens > max_tokens:
                chunks.append(" ".join(current_chunk)) # If it exceeds, finalize the current chunk and add to chunks list.
                current_chunk = [comment] # Start a new chunk with the current comment.
                current_chunk_tokens = tokens # Update token count for the new chunk.
            else:
                current_chunk.append(comment) # If it doesn't exceed, add the comment to the current chunk.
                current_chunk_tokens += tokens # Update token count for the current chunk.

    # Add any remaining comments in the current_chunk to the chunks list after processing all comments.
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Print the chunking results for debugging or monitoring.
    print(f"Total number of chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} token count: {get_token_count(chunk, tokenizer)}")

    return chunks # Return the list of comment chunks.

# Function to load the summarization model and tokenizer.
# @st.cache_resource caches the result, so model and tokenizer are loaded only once.
@st.cache_resource
def get_summarization_model_and_tokenizer():
    model_name = "knkarthick/MEETING_SUMMARY" # Specify the pre-trained summarization model name from Hugging Face Model Hub.
    tokenizer = AutoTokenizer.from_pretrained(model_name) # Load the tokenizer associated with the model.
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name) # Load the sequence-to-sequence language model for summarization.
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # Determine if CUDA (GPU) is available, otherwise use CPU.
    model.to(device) # Move the model to the selected device (GPU or CPU).
    return model, tokenizer, device # Return the model, tokenizer, and device.

# Function to preprocess comments and generate summaries if needed.
def preprocess_comments_and_summarize(feedback_data, comment_column, batch_size=32, max_length=75, min_length=30, max_tokens=1000, very_short_limit=30):
    print("Starting preprocessing and summarization...") # Print start message.

    # 1. Preprocess the comments using the preprocess_text function.
    # Apply preprocess_text function to each comment in the specified comment_column of the feedback_data DataFrame.
    feedback_data['preprocessed_comments'] = feedback_data[comment_column].apply(preprocess_text)
    print("Comments preprocessed.") # Print preprocessing completion message.

    # 2. Get the cached summarization model and tokenizer.
    model, tokenizer, device = get_summarization_model_and_tokenizer()
    print("Summarization model and tokenizer retrieved from cache.") # Print cache retrieval message.

    # 3. Separate comments into categories based on their token count.
    # very_short_comments: comments with token count less than or equal to very_short_limit.
    very_short_comments = [comment for comment in feedback_data['preprocessed_comments'].tolist() if get_token_count(comment, tokenizer) <= very_short_limit]
    # short_comments: comments with token count between very_short_limit and max_tokens (exclusive of very_short_limit, inclusive of max_tokens).
    short_comments = [comment for comment in feedback_data['preprocessed_comments'].tolist() if very_short_limit < get_token_count(comment, tokenizer) <= max_tokens]
    # long_comments: comments with token count greater than max_tokens.
    long_comments = [comment for comment in feedback_data['preprocessed_comments'].tolist() if get_token_count(comment, tokenizer) > max_tokens]
    print(f"Separated comments into categories: {len(very_short_comments)} very short, {len(short_comments)} short, and {len(long_comments)} long comments.") # Print comment category counts.

    # 4. Handle very short comments: no summarization needed, use original comment as summary.
    summaries_dict = {comment: comment for comment in very_short_comments} # Create a dictionary mapping very short comments to themselves (no summary).
    print(f"{len(very_short_comments)} very short comments directly added to summaries.") # Print message.

    # 5. Handle short comments: summarize in batches.
    pbar = tqdm(total=len(short_comments), desc="Summarizing short comments") # Initialize progress bar for summarization of short comments.
    for i in range(0, len(short_comments), batch_size): # Iterate through short comments in batches.
        batch = short_comments[i:i+batch_size] # Get a batch of short comments.
        # Summarize each comment in the batch using summarize_text function.
        summaries = [summarize_text(comment, tokenizer, model, device, max_length, min_length) for comment in batch]
        # Store the summaries in the summaries_dict, mapping original comment to its summary.
        for original_comment, summary in zip(batch, summaries):
            summaries_dict[original_comment] = summary
        pbar.update(len(batch)) # Update progress bar.
    pbar.close() # Close progress bar after processing all short comments.

    # 6. Handle long comments: split into chunks, summarize each chunk, then re-summarize if needed.
    pbar = tqdm(total=len(long_comments), desc="Summarizing long comments") # Initialize progress bar for long comments.
    for comment in long_comments: # Iterate through long comments.
        # Split the long comment into chunks using split_comments_into_chunks function.
        chunks = split_comments_into_chunks([(comment, get_token_count(comment, tokenizer))], tokenizer, max_tokens)
        # Summarize each chunk.
        summaries = [summarize_text(chunk, tokenizer, model, device, max_length, min_length) for chunk in chunks]
        full_summary = " ".join(summaries) # Join the summaries of all chunks to get a full summary of the long comment.

        resummarization_count = 0 # Initialize counter for re-summarization.
        # Re-summarize the full summary if it's still too long (token count > max_length).
        while get_token_count(full_summary, tokenizer) > max_length:
            resummarization_count += 1 # Increment re-summarization count.
            print(f"Re-summarizing a long comment with token count: {get_token_count(full_summary, tokenizer)}") # Print re-summarization message.
            full_summary = summarize_text(full_summary, tokenizer, model, device, max_length, min_length) # Re-summarize the full summary.

        # Display the number of times a comment was re-summarized if it was re-summarized.
        if resummarization_count > 0:
            print(f"Long comment was re-summarized {resummarization_count} times to fit the max length.")

        summaries_dict[comment] = full_summary # Store the final summary in summaries_dict.
        pbar.update(1) # Update progress bar.
    pbar.close() # Close progress bar after processing all long comments.

    print("Preprocessing and summarization completed.") # Print completion message.
    return summaries_dict # Return the dictionary containing summaries for all comments.


# Function to summarize a given text using the loaded summarization model.
def summarize_text(text, tokenizer, model, device, max_length, min_length):
    # Tokenize the input text for summarization model.
    input_ids = tokenizer([text], truncation=True, padding=True, return_tensors='pt')['input_ids'].to(device)
    # Generate summary using the model.
    summary_ids = model.generate(input_ids, max_length=max_length, min_length=min_length)[0]
    # Decode the summary token IDs back to text, skipping special tokens.
    return tokenizer.decode(summary_ids, skip_special_tokens=True)


# Function to compute semantic similarity between a comment embedding and a keyword embedding.
def compute_semantic_similarity(comment_embedding, keyword_embedding):
    # cosine_similarity from scikit-learn calculates the cosine similarity between two vectors.
    # It expects 2D arrays, so we pass embeddings as lists within lists: [[comment_embedding]], [[keyword_embedding]].
    # [0][0] extracts the single similarity score from the resulting 2D array.
    return cosine_similarity([comment_embedding], [keyword_embedding])[0][0]


# Set the default layout mode for Streamlit page to "wide" for better use of screen space.
st.set_page_config(layout="wide")

# Streamlit interface - Title of the application.
st.title("👨‍💻 Transcript Categorization")

# Initialize BERT model for sentence embeddings once and cache it for reuse.
model = initialize_bert_model()

# Add a checkbox in the sidebar for "Emerging Issue Mode".
emerging_issue_mode = st.sidebar.checkbox("Emerging Issue Mode")

# Sidebar description to explain "Emerging Issue Mode" to the user.
st.sidebar.write("Emerging issue mode allows you to set a minimum similarity score. If the comment doesn't match up to the categories based on the threshold, it will be set to NO MATCH.")

# Initialize variables for similarity threshold, score, and best match score.
similarity_threshold = None
similarity_score = None
best_match_score = None

# If "Emerging Issue Mode" is activated, display a slider to set the semantic similarity threshold.
if emerging_issue_mode:
    similarity_threshold = st.sidebar.slider("Semantic Similarity Threshold", min_value=0.0, max_value=1.0, value=0.35)

# Initialize an empty dictionary to hold categories (though it's not directly used here, default_categories is used instead).
categories = {}

# Sidebar section for editing categories, subcategories, and keywords.
st.sidebar.header("Edit Categories")

# Create a new dictionary to store the updated categories from user input.
new_categories = {}

# Iterate through the default categories and subcategories to create editable fields in the sidebar.
for category, subcategories in default_categories.items():
    # Create a text input field in the sidebar for each category name, pre-filled with the default category name.
    category_name = st.sidebar.text_input(f"{category} Category", value=category)

    # Create a new dictionary to store updated subcategories for the current category.
    new_subcategories = {}

    # Iterate through subcategories within the current category.
    for subcategory, keywords in subcategories.items():
        # Create a text input field for each subcategory name, pre-filled with the default subcategory name.
        subcategory_name = st.sidebar.text_input(f"{subcategory} Subcategory under {category_name}", value=subcategory)

        # Create an expander in the sidebar for keywords related to the current subcategory.
        with st.sidebar.expander(f"Keywords for {subcategory_name}"):
            # Create a text area for editing keywords, pre-filled with default keywords (newline separated).
            category_keywords = st.text_area("Keywords", value="\n".join(keywords))

        # Update the keywords in the new_subcategories dictionary by splitting the text area input by newlines.
        new_subcategories[subcategory_name] = category_keywords.split("\n")

    # Update the subcategories for the current category in the new_categories dictionary.
    new_categories[category_name] = new_subcategories

# Replace the original default_categories dictionary with the new_categories dictionary,
# effectively updating the categories based on user edits from the sidebar.
default_categories = new_categories

# File uploader widget in Streamlit to upload a CSV file. Allows only CSV files.
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

# Initialize variables to store column names, processed data, etc.
comment_column = None
date_column = None
trends_data = None
all_processed_data = []  # List to store processed data from each chunk

# Define an empty Pandas DataFrame to store feedback data.
feedback_data = pd.DataFrame()

# Process the uploaded file if a file has been uploaded.
if uploaded_file is not None:
    # Detect the encoding of the uploaded CSV file to handle different character sets correctly.
    csv_data = uploaded_file.read() # Read the file content as bytes.
    result = chardet.detect(csv_data) # Detect encoding using chardet library.
    encoding = result['encoding'] # Extract the detected encoding type.

    # Reset the file pointer to the beginning of the file so pandas can read it from the start.
    uploaded_file.seek(0)
    total_rows = sum(1 for row in uploaded_file) - 1  # Count total rows in the file, subtract 1 for header.

    # Calculate estimated total chunks for progress bar updates, assuming chunksize of 32.
    chunksize = 32  # Chunk size used for reading CSV in chunks later.
    estimated_total_chunks = math.ceil(total_rows / chunksize)

    try:
        # Read the first chunk (just one row) of the CSV to get the column names.
        first_chunk = next(pd.read_csv(BytesIO(csv_data), encoding=encoding, chunksize=1))
        column_names = first_chunk.columns.tolist() # Get the list of column names from the first chunk.
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}") # Display an error message if CSV reading fails.


    # UI elements for column selection - dropdowns to select comment and date columns.
    comment_column = st.selectbox("Select the column containing the comments", column_names)
    date_column = st.selectbox("Select the column containing the dates", column_names)

    # Grouping Options - Radio buttons to select date grouping for analysis.
    grouping_option = st.radio("Select how to group the dates", ["Date", "Week", "Month", "Quarter", "Hour"])
    process_button = st.button("Process Feedback") # Button to trigger the feedback processing.

    progress_bar = st.progress(0) # Initialize a progress bar to show processing progress.
    processed_chunks_count = 0 # Initialize counter for processed chunks.

    # Placeholders in Streamlit to display various outputs dynamically.
    trends_dataframe_placeholder = st.empty() # Placeholder for the main trends DataFrame.
    download_link_placeholder = st.empty() # Placeholder for the download link for Excel file.

    # Titles for different sections of the Streamlit app.
    st.subheader("All Categories Trends Line Chart") # Subheader for the line chart.
    line_chart_placeholder = st.empty() # Placeholder for the line chart.

    st.subheader("Pivot table for category trends") # Subheader for the pivot table.
    pivot_table_placeholder = st.empty() # Placeholder for the pivot table.

    st.subheader("Category vs Sentiment and Quantity") # Subheader for Category vs Sentiment section.
    category_sentiment_dataframe_placeholder = st.empty() # Placeholder for category sentiment DataFrame.
    category_sentiment_bar_chart_placeholder = st.empty() # Placeholder for category sentiment bar chart.

    st.subheader("Sub-Category vs Sentiment and Quantity") # Subheader for Sub-Category vs Sentiment section.
    subcategory_sentiment_dataframe_placeholder = st.empty() # Placeholder for subcategory sentiment DataFrame.
    subcategory_sentiment_bar_chart_placeholder = st.empty() # Placeholder for subcategory sentiment bar chart.

    st.subheader("Top 10 Most Recent Comments for Each Top Subcategory") # Subheader for top comments section.
    # Create combined placeholders for top comments' titles and tables for 10 subcategories.
    combined_placeholders = [(st.empty(), st.empty()) for _ in range(10)]


    # Function to process feedback data, categorize comments, perform sentiment analysis, and generate trends data.
    # @st.cache_data(persist="disk") caches the output of this function based on input arguments, persisting to disk.
    @st.cache_data(persist="disk")
    def process_feedback_data(feedback_data, comment_column, date_column, categories, similarity_threshold):
        global previous_categories # Access the global variable previous_categories for caching logic.

        # Retrieve the cached keyword embeddings using the compute_keyword_embeddings function.
        keyword_embeddings = compute_keyword_embeddings(categories)

        # Check if the categories have changed since the last computation of embeddings.
        if previous_categories != categories:
            keyword_embeddings = compute_keyword_embeddings(categories) # Recompute keyword embeddings if categories have changed.
            previous_categories = categories.copy() # Update previous_categories to the current categories.
        else:
            # If categories haven't changed, and embeddings are not yet computed (shouldn't happen due to caching), compute them.
            if not keyword_embeddings:
                keyword_embeddings = compute_keyword_embeddings(categories)

        # Initialize empty lists to store processed comment information.
        categorized_comments = []
        sentiments = []
        similarity_scores = []
        summarized_texts = []
        categories_list = []

        # Initialize the BERT model once - already initialized globally, so this line is commented out.
        #model = initialize_bert_model()


        # Preprocess comments and summarize if necessary using the preprocess_comments_and_summarize function.
        start_time = time.time()
        print("def process_feedback_data:Preprocessing comments and summarizing if necessary...")

        summaries_dict = preprocess_comments_and_summarize(feedback_data, comment_column)

        # Create a new column 'preprocessed_comments' by applying preprocess_text to the original comment column.
        feedback_data['preprocessed_comments'] = feedback_data[comment_column].apply(preprocess_text)
        # Create a new column 'summarized_comments' by mapping preprocessed comments to their summaries from summaries_dict.
        feedback_data['summarized_comments'] = feedback_data['preprocessed_comments'].map(summaries_dict)

        # Fill any missing values in 'summarized_comments' (if any) with the original 'preprocessed_comments'.
        # This ensures that even if summarization failed for some reason, we still have the preprocessed comment.
        feedback_data['summarized_comments'] = feedback_data['summarized_comments'].fillna(feedback_data['preprocessed_comments'])

        end_time = time.time()
        print(f"Preprocessed comments and summarized. Time taken: {end_time - start_time} seconds.")


        # Compute comment embeddings in batches for efficiency.
        start_time = time.time()
        print("Start comment embeddings in batches")
        batch_size = 1024  # Define batch size for embedding computation.
        comment_embeddings = [] # Initialize list to store comment embeddings.
        for i in range(0, len(feedback_data), batch_size): # Iterate through feedback data in batches.
            batch = feedback_data['summarized_comments'][i:i+batch_size].tolist() # Get a batch of summarized comments.
            comment_embeddings.extend(model.encode(batch)) # Compute embeddings for the batch and extend the embeddings list.
        feedback_data['comment_embeddings'] = comment_embeddings # Assign the computed embeddings to a new column in the DataFrame.
        end_time = time.time()
        print(f"Batch comment embeddings done. Time taken: {end_time - start_time} seconds.")

        # Compute sentiment scores for preprocessed comments.
        start_time = time.time()
        print("Computing sentiment scores...")
        feedback_data['sentiment_scores'] = feedback_data['preprocessed_comments'].apply(perform_sentiment_analysis) # Apply sentiment analysis function.
        end_time = time.time()
        print(f"Sentiment scores computed. Time taken: {end_time - start_time} seconds.")

        # Compute semantic similarity and assign categories in batches.
        start_time = time.time()
        print("Computing semantic similarity and assigning categories...")
        # Initialize lists to store category assignment results.
        categories_list = [''] * len(feedback_data)
        sub_categories_list = [''] * len(feedback_data)
        keyphrases_list = [''] * len(feedback_data)
        summarized_texts = [''] * len(feedback_data)
        similarity_scores = [0.0] * len(feedback_data)
        for i in range(0, len(feedback_data), batch_size): # Iterate through feedback data in batches.
            batch_embeddings = feedback_data['comment_embeddings'][i:i + batch_size].tolist() # Get a batch of comment embeddings.
            # Iterate through each keyword embedding in the precomputed keyword_embeddings dictionary.
            for (category, subcategory, keyword), embeddings in keyword_embeddings.items():
                # Compute semantic similarity between the batch of comment embeddings and the current keyword embedding.
                batch_similarity_scores = [compute_semantic_similarity(batch_embedding, embeddings) for batch_embedding in batch_embeddings]
                # Update categories, sub-categories, and keyphrases based on the highest similarity score found so far for each comment in the batch.
                for j, similarity_score in enumerate(batch_similarity_scores):
                    idx = i + j  # Calculate the original index of the comment in the full DataFrame.
                    if idx < len(categories_list): # Check if the index is within the valid range.
                        if similarity_score > similarity_scores[idx]: # If current similarity score is higher than the existing highest score for this comment.
                            categories_list[idx] = category # Update category.
                            sub_categories_list[idx] = subcategory # Update sub-category.
                            keyphrases_list[idx] = keyword # Update keyphrase.
                            summarized_texts[idx] = keyword # Update summarized text (using keyword for now, might need to refine).
                            similarity_scores[idx] = similarity_score # Update highest similarity score.
                    else: # This branch might be redundant as idx should always be within range in this loop structure.
                        categories_list.append(category)
                        sub_categories_list.append(subcategory)
                        keyphrases_list.append(keyword)
                        summarized_texts.append(keyword)
                        similarity_scores.append(similarity_score)

        end_time = time.time()
        print(f"Computed semantic similarity and assigned categories. Time taken: {end_time - start_time} seconds.")

        # After category matching is done, drop the 'comment_embeddings' column to save memory as it's no longer needed.
        feedback_data.drop(columns=['comment_embeddings'], inplace=True)


        # Prepare the final processed data by iterating through each row of the feedback_data DataFrame.
        categorized_comments = []
        for index in range(len(feedback_data)):
            row = feedback_data.iloc[index]  # Get each row as a Series.

            preprocessed_comment = row['preprocessed_comments'] # Extract preprocessed comment.
            sentiment_score = row['sentiment_scores'] # Extract sentiment score.
            category = categories_list[index] # Get assigned category from the categories_list.
            sub_category = sub_categories_list[index] # Get assigned sub-category.
            keyphrase = keyphrases_list[index] # Get matched keyphrase.
            best_match_score = similarity_scores[index] # Get the best match similarity score.
            summarized_text = row['summarized_comments'] # Extract the summarized text.

            # Emerging Issue Mode Logic: If enabled and best match score is below threshold, categorize as "No Match".
            if emerging_issue_mode and best_match_score < similarity_threshold:
                category = 'No Match'
                sub_category = 'No Match'
                #keyphrase = 'No Match' #commented out so that keyphrase will be set to nearest match for analysis, even if category is "No Match".

            parsed_date = row[date_column].split(' ')[0] if isinstance(row[date_column], str) else None # Extract date part from date column string.
            # Extract the 'hour' from 'Parsed Date'.
            hour = pd.to_datetime(row[date_column]).hour # Convert date column to datetime and extract hour.

            # Extend the row data with processed information.
            row_extended = row.tolist() + [preprocessed_comment, summarized_text, category, sub_category, keyphrase, sentiment_score, best_match_score, parsed_date, hour]
            categorized_comments.append(row_extended) # Append the extended row to the list.

        # Create a new DataFrame 'trends_data' from the processed and categorized comments.
        existing_columns = feedback_data.columns.tolist() # Get original column names.
        additional_columns = [comment_column, 'Summarized Text', 'Category', 'Sub-Category', 'Keyphrase', 'Sentiment', 'Best Match Score', 'Parsed Date', 'Hour'] # Define new column names.
        headers = existing_columns + additional_columns # Combine original and new column names for DataFrame headers.
        trends_data = pd.DataFrame(categorized_comments, columns=headers) # Create DataFrame with processed data and headers.

        # Handle duplicate column names that might arise due to adding comment_column again as 'Summarized Text'.
        trends_data = trends_data.loc[:, ~trends_data.columns.duplicated()] # Select columns, excluding duplicates (keeps first occurrence).
        duplicate_columns = set([col for col in trends_data.columns if trends_data.columns.tolist().count(col) > 1]) # Find columns with duplicates.
        for column in duplicate_columns: # Iterate through duplicate column names.
            column_indices = [i for i, col in enumerate(trends_data.columns) if col == column] # Get indices of duplicate columns.
            for i, idx in enumerate(column_indices[1:], start=1): # Iterate through duplicate column indices, starting from the second one.
                trends_data.columns.values[idx] = f"{column}_{i}" # Rename duplicate columns by adding _1, _2, etc.

        return trends_data # Return the final processed DataFrame.

    # Main processing logic that runs when 'Process Feedback' button is clicked.
    if comment_column is not None and date_column is not None and grouping_option is not None and process_button:
        chunk_iter = pd.read_csv(BytesIO(csv_data), encoding=encoding, chunksize=32)  # Read CSV in chunks for memory efficiency.

        # Initialize a list to store processed chunks of data.
        processed_chunks = []

        # Iterate through each chunk of the CSV data.
        for feedback_data in chunk_iter:

            processed_chunk = process_feedback_data(feedback_data, comment_column, date_column, default_categories, similarity_threshold) # Process each chunk.
            processed_chunks.append(processed_chunk) # Append the processed chunk to the list.

            # Concatenate all processed chunks into a single DataFrame 'trends_data'.
            trends_data = pd.concat(processed_chunks, ignore_index=True)


            # Now, perform analysis and display results on the cumulative 'trends_data'.
            # Display the processed trends data in a DataFrame in Streamlit.
            if trends_data is not None:
                #st.title("Feedback Trends and Insights")
                trends_dataframe_placeholder.dataframe(trends_data)

                # Display pivot table with counts for Category, Sub-Category, and Parsed Date
                #st.subheader("All Categories Trends")

                # Convert 'Parsed Date' to datetime format if it's not already.
                trends_data['Parsed Date'] = pd.to_datetime(trends_data['Parsed Date'], errors='coerce')
                # Extract hour from 'Parsed Date' - already done in process_feedback_data, but ensuring here.

                # Create pivot table to analyze trends over time based on the selected grouping option.
                if grouping_option == 'Date':
                    pivot = trends_data.pivot_table(
                        index=['Category', 'Sub-Category'], # Rows of the pivot table are Category and Sub-Category.
                        columns=pd.Grouper(key='Parsed Date', freq='D'), # Columns are dates, grouped daily ('D').
                        values='Sentiment', # Values in the table are based on 'Sentiment' column.
                        aggfunc='count', # Aggregate function is count (number of comments).
                        fill_value=0 # Fill missing values with 0.
                    )

                elif grouping_option == 'Week':
                    pivot = trends_data.pivot_table(
                        index=['Category', 'Sub-Category'],
                        columns=pd.Grouper(key='Parsed Date', freq='W-SUN', closed='left', label='left'), # Weekly grouping, week starts on Sunday.
                        values='Sentiment',
                        aggfunc='count',
                        fill_value=0
                    )

                elif grouping_option == 'Month':
                    pivot = trends_data.pivot_table(
                        index=['Category', 'Sub-Category'],
                        columns=pd.Grouper(key='Parsed Date', freq='M'), # Monthly grouping.
                        values='Sentiment',
                        aggfunc='count',
                        fill_value=0
                    )
                elif grouping_option == 'Quarter':
                    pivot = trends_data.pivot_table(
                        index=['Category', 'Sub-Category'],
                        columns=pd.Grouper(key='Parsed Date', freq='Q'), # Quarterly grouping.
                        values='Sentiment',
                        aggfunc='count',
                        fill_value=0
                    )
                elif grouping_option == 'Hour':
                    if 'Hour' not in trends_data.columns: # Check if 'Hour' column exists, if not, extract it.
                        print("Hour column not found in trends_data. Extracting now...")
                        # Ensure the date column is in datetime format
                        feedback_data[date_column] = pd.to_datetime(feedback_data[date_column])
                        # Extract 'Hour' from 'Parsed Date' and add it to the DataFrame
                        trends_data['Hour'] = feedback_data[date_column].dt.hour
                    else:
                        print("Hour column already exists in trends_data.")

                    pivot = trends_data.pivot_table(
                        index=['Category', 'Sub-Category'],
                        columns='Hour',  # Use the 'Hour' column for pivot table
                        values='Sentiment',
                        aggfunc='count',
                        fill_value=0
                    )
                    # Convert the 'Hour' column names to datetime objects to format x-axis nicely in charts.
                    pivot.columns = pd.to_datetime(pivot.columns, format='%H').time

                pivot.columns = pivot.columns.astype(str)  # Convert column labels (dates/hours) to strings for display.

                # Sort the pivot table rows based on the total count (sum across columns) in descending order.
                pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

                # Sort the pivot table columns (dates/hours) in descending order based on most recent date/hour.
                pivot = pivot[sorted(pivot.columns, reverse=True)]

                # Create a line chart for the top 5 trends over time with the selected grouping option.
                # Reset index to make 'Category' and 'Sub-Category' regular columns instead of index.
                pivot_reset = pivot.reset_index()

                # Set 'Sub-Category' as the index for easier plotting of trends.
                pivot_reset = pivot_reset.set_index('Sub-Category')

                # Drop the 'Category' column as we are focusing on sub-category trends in the chart.
                pivot_reset = pivot_reset.drop(columns=['Category'])

                # Get the top 5 sub-categories based on their trend counts.
                top_5_trends = pivot_reset.head(5).T  # Transpose DataFrame to have dates/hours as index for line chart.

                # Display the line chart in Streamlit using the placeholder.
                line_chart_placeholder.line_chart(top_5_trends)

                # Display the pivot table in Streamlit using the placeholder.
                pivot_table_placeholder.dataframe(pivot)

                # Create pivot table to show Category vs Sentiment and Quantity.
                pivot1 = trends_data.groupby('Category')['Sentiment'].agg(['mean', 'count']) # Group by Category and aggregate Sentiment by mean and count.
                pivot1.columns = ['Average Sentiment', 'Quantity'] # Rename columns of pivot table.
                pivot1 = pivot1.sort_values('Quantity', ascending=False) # Sort by Quantity in descending order.

                # Create pivot table to show Sub-Category vs Sentiment and Quantity.
                pivot2 = trends_data.groupby(['Category', 'Sub-Category'])['Sentiment'].agg(['mean', 'count']) # Group by Category and Sub-Category.
                pivot2.columns = ['Average Sentiment', 'Quantity'] # Rename columns.
                pivot2 = pivot2.sort_values('Quantity', ascending=False) # Sort by Quantity.

                # Reset index for pivot2 to make 'Sub-Category' a regular column again.
                pivot2_reset = pivot2.reset_index()


                # Set 'Sub-Category' as the index for easier bar chart plotting.
                pivot2_reset.set_index('Sub-Category', inplace=True)

                # Create and display a bar chart for Category Quantity using the placeholder.
                category_sentiment_bar_chart_placeholder.bar_chart(pivot1['Quantity'])

                # Display the Category vs Sentiment and Quantity DataFrame in Streamlit.
                #st.subheader("Category vs Sentiment and Quantity")
                category_sentiment_dataframe_placeholder.dataframe(pivot1)

                # Create and display a bar chart for Sub-Category Quantity using the placeholder.
                subcategory_sentiment_bar_chart_placeholder.bar_chart(pivot2_reset['Quantity'])

                # Display the Sub-Category vs Sentiment and Quantity DataFrame in Streamlit.
                #st.subheader("Sub-Category vs Sentiment and Quantity")
                subcategory_sentiment_dataframe_placeholder.dataframe(pivot2_reset)

                # Display top 10 most recent comments for each of the top 10 subcategories.
                #st.subheader("Top 10 Most Recent Comments for Each Top Subcategory")

                # Get the top 10 subcategories based on the survey count from pivot2_reset.
                top_subcategories = pivot2_reset.head(10).index.tolist()

                # Update the subheader titles and display top comments table for each of the top 10 subcategories.
                for idx, subcategory in enumerate(top_subcategories):
                    # Extract the title and table placeholders for the current subcategory.
                    title_placeholder, table_placeholder = combined_placeholders[idx]

                    # Update the subheader title placeholder with the subcategory name.
                    title_placeholder.subheader(subcategory)

                    # Filter the trends_data DataFrame to get data only for the current subcategory.
                    filtered_data = trends_data[trends_data['Sub-Category'] == subcategory]

                    # Get the top 10 most recent comments for the current subcategory, sorted by 'Parsed Date' descending.
                    top_comments = filtered_data.nlargest(10, 'Parsed Date')[['Parsed Date', comment_column, 'Summarized Text', 'Keyphrase', 'Sentiment', 'Best Match Score']]

                    # Format the 'Parsed Date' column to display only the date part as string.
                    top_comments['Parsed Date'] = top_comments['Parsed Date'].dt.date.astype(str)

                    # Display the top comments in a table using the placeholder.
                    table_placeholder.table(top_comments)

                # Format 'Parsed Date' column in trends_data to string in 'YYYY-MM-DD' format for Excel export and pivot tables.
                trends_data['Parsed Date'] = trends_data['Parsed Date'].dt.strftime('%Y-%m-%d').fillna('')

                # Create pivot table for Excel export, grouping by Category, Sub-Category, and Parsed Date.
                pivot = trends_data.pivot_table(
                    index=['Category', 'Sub-Category'],
                    columns=pd.to_datetime(trends_data['Parsed Date']).dt.strftime('%Y-%m-%d'),  # Format column headers as 'YYYY-MM-DD'.
                    values='Sentiment',
                    aggfunc='count',
                    fill_value=0
                )

                # Sort rows and columns of the pivot table for Excel export similar to display pivot table.
                pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
                pivot = pivot[sorted(pivot.columns, reverse=True)]

            # Update the progress bar after processing each chunk.
            processed_chunks_count += 1
            progress_value = processed_chunks_count / estimated_total_chunks  # Calculate progress as fraction of processed chunks.
            progress_bar.progress(progress_value) # Update the Streamlit progress bar.




            # Save the processed DataFrame and pivot tables to an Excel file in memory.
            excel_file = BytesIO() # Create an in-memory binary stream to store Excel file.
            with pd.ExcelWriter(excel_file, engine='xlsxwriter', mode='xlsx') as excel_writer: # Use pandas ExcelWriter with xlsxwriter engine.
                trends_data.to_excel(excel_writer, sheet_name='Feedback Trends and Insights', index=False) # Write trends_data to Excel sheet.

                # Convert 'Parsed Date' column to datetime type for correct Excel formatting and grouping.
                trends_data['Parsed Date'] = pd.to_datetime(trends_data['Parsed Date'], errors='coerce')

                # Create a separate column 'Formatted Date' with date in 'YYYY-MM-DD' format for pivot table indexing in Excel.
                trends_data['Formatted Date'] = trends_data['Parsed Date'].dt.strftime('%Y-%m-%d')

                # Handle potential 'level_0' column that might be created during DataFrame operations.
                if 'level_0' in trends_data.columns:
                    trends_data.drop(columns='level_0', inplace=True) # Drop 'level_0' if it exists.

                # Reset index of trends_data to make index a regular column before setting 'Formatted Date' as index.
                trends_data.reset_index(inplace=True)

                # Set 'Formatted Date' column as the index for the DataFrame in Excel.
                trends_data.set_index('Formatted Date', inplace=True)

                # Create pivot table for Excel export, grouped by selected grouping_option.
                if grouping_option == 'Date':
                    pivot = trends_data.pivot_table(
                        index=['Category', 'Sub-Category'],
                        columns='Parsed Date', # Use 'Parsed Date' column for daily pivot.
                        values='Sentiment',
                        aggfunc='count',
                        fill_value=0
                    )
                elif grouping_option == 'Week':
                    pivot = trends_data.pivot_table(
                        index=['Category', 'Sub-Category'],
                        columns=pd.Grouper(key='Parsed Date', freq='W-SUN', closed='left', label='left'), # Weekly pivot.
                        values='Sentiment',
                        aggfunc='count',
                        fill_value=0
                    )

                elif grouping_option == 'Month':
                    pivot = trends_data.pivot_table(
                        index=['Category', 'Sub-Category'],
                        columns=pd.Grouper(key='Parsed Date', freq='M'), # Monthly pivot.
                        values='Sentiment',
                        aggfunc='count',
                        fill_value=0
                    )
                elif grouping_option == 'Quarter':
                    pivot = trends_data.pivot_table(
                        index=['Category', 'Sub-Category'],
                        columns=pd.Grouper(key='Parsed Date', freq='Q'), # Quarterly pivot.
                        values='Sentiment',
                        aggfunc='count',
                        fill_value=0
                    )
                elif grouping_option == 'Hour':
                    # Ensure the date column is in datetime format - already done before, but ensuring again.
                    feedback_data[date_column] = pd.to_datetime(feedback_data[date_column])
                    pivot = trends_data.pivot_table(
                        index=['Category', 'Sub-Category'],
                        columns=feedback_data[date_column].dt.hour, # Hourly pivot.
                        values='Sentiment',
                        aggfunc='count',
                        fill_value=0
                    )

                # Format column headers of pivot table as date strings in 'YYYY-MM-DD' format for Excel sheet.
                if grouping_option != 'Hour':
                    pivot.columns = pivot.columns.strftime('%Y-%m-%d')



                # Write pivot tables to different sheets in the Excel file.
                pivot.to_excel(excel_writer, sheet_name='Trends by ' + grouping_option, merge_cells=False) # Sheet for trend pivot.
                pivot1.to_excel(excel_writer, sheet_name='Categories', merge_cells=False) # Sheet for category summary.
                pivot2.to_excel(excel_writer, sheet_name='Subcategories', merge_cells=False) # Sheet for subcategory summary.

                # Write example comments for top subcategories to a separate sheet in Excel.
                example_comments_sheet = excel_writer.book.add_worksheet('Example Comments') # Create a new worksheet.

                # Write each table of example comments to the 'Example Comments' sheet.
                for subcategory in top_subcategories:
                    filtered_data = trends_data[trends_data['Sub-Category'] == subcategory] # Filter data for current subcategory.
                    top_comments = filtered_data.nlargest(10, 'Parsed Date')[['Parsed Date', comment_column]] # Get top 10 recent comments.
                    # Calculate starting row for each subcategory's table in the Excel sheet.
                    start_row = (top_subcategories.index(subcategory) * 8) + 1

                    # Write the subcategory name as a merged cell as a title for the table.
                    example_comments_sheet.merge_range(start_row, 0, start_row, 1, subcategory)
                    example_comments_sheet.write(start_row, 2, '') # Add an empty cell for spacing.
                    # Write table headers for Date and Comment Column.
                    example_comments_sheet.write(start_row + 1, 0, 'Date')
                    example_comments_sheet.write(start_row + 1, 1, comment_column)

                    # Write the top comments data to the Excel sheet.
                    for i, (_, row) in enumerate(top_comments.iterrows(), start=start_row + 2):
                        example_comments_sheet.write(i, 0, row['Parsed Date']) # Write date.
                        example_comments_sheet.write_string(i, 1, str(row[comment_column])) # Write comment, ensuring it's treated as a string.

                # Save the Excel file to the in-memory stream.
            if not excel_writer.book.fileclosed:
                excel_writer.close()

            # Convert the in-memory Excel file to bytes and create a download link in Streamlit.
            excel_file.seek(0) # Reset stream position to the beginning.
            b64 = base64.b64encode(excel_file.read()).decode() # Encode Excel file bytes to base64 string.
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="feedback_trends.xlsx">Download Excel File</a>' # Create HTML download link.
            download_link_placeholder.markdown(href, unsafe_allow_html=True) # Display download link in Streamlit.
