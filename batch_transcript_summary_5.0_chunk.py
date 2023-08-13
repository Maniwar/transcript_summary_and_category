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
@st.cache_data(persist="disk")
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

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

# Select the column containing the comments
comment_column = None
date_column = None
trends_data = None
all_processed_data = []  # List to store processed data from each chunk

# Define an empty DataFrame for feedback_data
feedback_data = pd.DataFrame()

if uploaded_file is not None:
    # Detect the encoding of the CSV file
    csv_data = uploaded_file.read()
    result = chardet.detect(csv_data)
    encoding = result['encoding']
    
    # Reset the file pointer to the beginning and count the number of rows
    uploaded_file.seek(0)
    total_rows = sum(1 for row in uploaded_file) - 1  # Subtract 1 for the header
    
    # Calculate estimated total chunks
    chunksize = 32  # This is the chunksize you've set in your code
    estimated_total_chunks = math.ceil(total_rows / chunksize)
    
    try:
        # Read the first chunk to get the column names
        first_chunk = next(pd.read_csv(BytesIO(csv_data), encoding=encoding, chunksize=1))
        column_names = first_chunk.columns.tolist()
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")


    # UI elements for column selection
    comment_column = st.selectbox("Select the column containing the comments", column_names)
    date_column = st.selectbox("Select the column containing the dates", column_names)

    # Grouping Options
    grouping_option = st.radio("Select how to group the dates", ["Date", "Week", "Month", "Quarter", "Hour"])
    process_button = st.button("Process Feedback")
    
    progress_bar = st.progress(0)
    processed_chunks_count = 0

    def process_feedback_data(feedback_data, comment_column, date_column, categories, similarity_threshold):
        global previous_categories
        
        # Define keyword_embeddings at the top
        keyword_embeddings = {}
        
        # Check if we already computed embeddings for these categories
        if previous_categories != categories:
            keyword_embeddings = compute_keyword_embeddings(categories)
            previous_categories = categories.copy()
        else:
            # If the embeddings aren't computed yet, compute them (this is for the initial run)
            if not keyword_embeddings:
                keyword_embeddings = compute_keyword_embeddings(categories)


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
        print("def process_feedback_data:Preprocessing comments and summarizing if necessary...")

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

            parsed_date = row[date_column].split(' ')[0] if isinstance(row[date_column], str) else None
            # Extract the 'hour' from 'Parsed Date'
            hour = pd.to_datetime(row[date_column]).hour

            row_extended = row.tolist() + [preprocessed_comment, summarized_text, category, sub_category, keyphrase, sentiment_score, best_match_score, parsed_date, hour]
            categorized_comments.append(row_extended)

        # Create a new DataFrame with extended columns
        existing_columns = feedback_data.columns.tolist()
        additional_columns = [comment_column, 'Summarized Text', 'Category', 'Sub-Category', 'Keyphrase', 'Sentiment', 'Best Match Score', 'Parsed Date', 'Hour']
        headers = existing_columns + additional_columns
        trends_data = pd.DataFrame(categorized_comments, columns=headers)

        # Rename duplicate column names
        trends_data = trends_data.loc[:, ~trends_data.columns.duplicated()]
        duplicate_columns = set([col for col in trends_data.columns if trends_data.columns.tolist().count(col) > 1])
        for column in duplicate_columns:
            column_indices = [i for i, col in enumerate(trends_data.columns) if col == column]
            for i, idx in enumerate(column_indices[1:], start=1):
                trends_data.columns.values[idx] = f"{column}_{i}"

        return trends_data

    if comment_column is not None and date_column is not None and grouping_option is not None and process_button:
        chunk_iter = pd.read_csv(BytesIO(csv_data), encoding=encoding, chunksize=64)  # Adjust chunksize as needed
        
        # Initialize a DataFrame to store the cumulative results
        cumulative_data = pd.DataFrame()

        for feedback_data in chunk_iter:
            processed_chunk = process_feedback_data(feedback_data, comment_column, date_column, default_categories, similarity_threshold)
            
            # Append the processed chunk to the cumulative DataFrame
            cumulative_data = pd.concat([cumulative_data, processed_chunk], ignore_index=True)
            
            # Update trends_data
            trends_data = cumulative_data

            # Now, do all the operations on the cumulative data
            # Display trends and insights
            if trends_data is not None:
                st.title("Feedback Trends and Insights")
                st.dataframe(trends_data)

                # Display pivot table with counts for Category, Sub-Category, and Parsed Date
                st.subheader("All Categories Trends")

                # Convert 'Parsed Date' into datetime format if it's not
                trends_data['Parsed Date'] = pd.to_datetime(trends_data['Parsed Date'], errors='coerce')
                # Extract hour from 'Parsed Date'

                # Create pivot table with counts for Category, Sub-Category, and Parsed Date
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
                        insdex=['Category', 'Sub-Category'],
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
                    # Ensure the date column is in datetime format
                    feedback_data[date_column] = pd.to_datetime(feedback_data[date_column])
                    # Extract 'Hour' from 'Parsed Date' and add it to the DataFrame
                    trends_data['Hour'] = feedback_data[date_column].dt.hour
                    pivot = trends_data.pivot_table(
                        index=['Category', 'Sub-Category'],
                        columns='Hour',  # Use the 'Hour' column for pivot table
                        values='Sentiment',
                        aggfunc='count',
                        fill_value=0
                    )
                    # Convert the 'Hour' column names to datetime objects
                    pivot.columns = pd.to_datetime(pivot.columns, format='%H').time

                pivot.columns = pivot.columns.astype(str)  # Convert column labels to strings

                # Sort the pivot table rows based on the highest count
                pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

                # Sort the pivot table columns in descending order based on the most recent date
                pivot = pivot[sorted(pivot.columns, reverse=True)]

                # Create a line chart for the top 5 trends over time with the selected grouping option
                # First, reset the index to have 'Category' and 'Sub-Category' as columns
                pivot_reset = pivot.reset_index()

                # Then, set 'Sub-Category' as the new index
                pivot_reset = pivot_reset.set_index('Sub-Category')

                # Drop the 'Category' column
                pivot_reset = pivot_reset.drop(columns=['Category'])

                # Now, get the top 5 trends
                top_5_trends = pivot_reset.head(5).T  # Transpose the DataFrame to have dates as index

                # Create and display a line chart for the top 5 trends
                st.line_chart(top_5_trends)

                # Display pivot table with counts for Category, Sub-Category, and Parsed Date
                st.dataframe(pivot)

                # Create pivot tables with counts
                pivot1 = trends_data.groupby('Category')['Sentiment'].agg(['mean', 'count'])
                pivot1.columns = ['Average Sentiment', 'Quantity']
                pivot1 = pivot1.sort_values('Quantity', ascending=False)

                pivot2 = trends_data.groupby(['Category', 'Sub-Category'])['Sentiment'].agg(['mean', 'count'])
                pivot2.columns = ['Average Sentiment', 'Quantity']
                pivot2 = pivot2.sort_values('Quantity', ascending=False)

                # Reset index for pivot2
                pivot2_reset = pivot2.reset_index()


                # Set 'Sub-Category' as the index
                pivot2_reset.set_index('Sub-Category', inplace=True)

                # Create and display a bar chart for pivot1 with counts
                st.bar_chart(pivot1['Quantity'])

                # Display pivot table with counts for Category
                st.subheader("Category vs Sentiment and Quantity")
                st.dataframe(pivot1)

                # Create and display a bar chart for pivot2 with counts
                st.bar_chart(pivot2_reset['Quantity'])

                # Display pivot table with counts for Sub-Category
                st.subheader("Sub-Category vs Sentiment and Quantity")
                st.dataframe(pivot2_reset)

                # Display top 10 most recent comments for each of the 10 top subcategories
                st.subheader("Top 10 Most Recent Comments for Each Top Subcategory")

                # Get the top 10 subcategories based on the survey count
                top_subcategories = pivot2_reset.head(10).index.tolist()

                # Iterate over the top subcategories
                for subcategory in top_subcategories:
                    st.subheader(subcategory)

                    # Filter the trends_data DataFrame for the current subcategory
                    filtered_data = trends_data[trends_data['Sub-Category'] == subcategory]

                    # Get the top 10 most recent comments for the current subcategory
                    top_comments = filtered_data.nlargest(10, 'Parsed Date')[['Parsed Date', comment_column,'Summarized Text','Sentiment', 'Best Match Score']]

                    # Format the parsed date to display only the date part
                    top_comments['Parsed Date'] = top_comments['Parsed Date'].dt.date.astype(str)

                    # Display the top comments as a table
                    st.table(top_comments)

                # Format 'Parsed Date' as string with 'YYYY-MM-DD' format
                trends_data['Parsed Date'] = trends_data['Parsed Date'].dt.strftime('%Y-%m-%d').fillna('')

                # Create pivot table with counts for Category, Sub-Category, and Parsed Date
                pivot = trends_data.pivot_table(
                    index=['Category', 'Sub-Category'],
                    columns=pd.to_datetime(trends_data['Parsed Date']).dt.strftime('%Y-%m-%d'),  # Format column headers as 'YYYY-MM-DD'
                    values='Sentiment',
                    aggfunc='count',
                    fill_value=0
                )

                # Sort the pivot table rows based on the highest count
                pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

                # Sort the pivot table columns in descending order based on the most recent date
                pivot = pivot[sorted(pivot.columns, reverse=True)]
            
            # Update the progress bar
            processed_chunks_count += 1
            progress_value = processed_chunks_count / estimated_total_chunks  # you need to estimate total chunks beforehand
            progress_bar.progress(progress_value)

            


            # Save DataFrame and pivot tables to Excel
            excel_file = BytesIO()
            with pd.ExcelWriter(excel_file, engine='xlsxwriter', mode='xlsx') as excel_writer:
                trends_data.to_excel(excel_writer, sheet_name='Feedback Trends and Insights', index=False)

                # Convert 'Parsed Date' column to datetime type
                trends_data['Parsed Date'] = pd.to_datetime(trends_data['Parsed Date'], errors='coerce')

                # Create a separate column for formatted date strings
                trends_data['Formatted Date'] = trends_data['Parsed Date'].dt.strftime('%Y-%m-%d')

                # Reset the index
                trends_data.reset_index(inplace=True)

                # Set 'Formatted Date' column as the index
                trends_data.set_index('Formatted Date', inplace=True)

                # Create pivot table with counts for Category, Sub-Category, and Parsed Date
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
                    # Ensure the date column is in datetime format
                    feedback_data[date_column] = pd.to_datetime(feedback_data[date_column])
                    pivot = trends_data.pivot_table(
                        index=['Category', 'Sub-Category'],
                        columns=feedback_data[date_column].dt.hour,
                        values='Sentiment',
                        aggfunc='count',
                        fill_value=0
                    )

                # Format column headers as date strings in 'YYYY-MM-DD' format
                if grouping_option != 'Hour':
                    pivot.columns = pivot.columns.strftime('%Y-%m-%d')

                # Write pivot tables to Excel
                pivot.to_excel(excel_writer, sheet_name='Trends by ' + grouping_option, merge_cells=False)
                pivot1.to_excel(excel_writer, sheet_name='Categories', merge_cells=False)
                pivot2.to_excel(excel_writer, sheet_name='Subcategories', merge_cells=False)

                # Write example comments to a single sheet
                example_comments_sheet = excel_writer.book.add_worksheet('Example Comments')

                # Write each table of example comments to the sheet
                for subcategory in top_subcategories:
                    filtered_data = trends_data[trends_data['Sub-Category'] == subcategory]
                    top_comments = filtered_data.nlargest(10, 'Parsed Date')[['Parsed Date', comment_column]]
                    # Calculate the starting row for each table
                    start_row = (top_subcategories.index(subcategory) * 8) + 1

                    # Write the subcategory as a merged cell
                    example_comments_sheet.merge_range(start_row, 0, start_row, 1, subcategory)
                    example_comments_sheet.write(start_row, 2, '')
                    # Write the table headers
                    example_comments_sheet.write(start_row + 1, 0, 'Date')
                    example_comments_sheet.write(start_row + 1, 1, comment_column)

                    # Write the table data
                    for i, (_, row) in enumerate(top_comments.iterrows(), start=start_row + 2):
                        example_comments_sheet.write(i, 0, row['Parsed Date'])
                        example_comments_sheet.write_string(i, 1, str(row[comment_column]))

                # Save the Excel file
                excel_writer.close()

            # Convert the Excel file to bytes and create a download link
            excel_file.seek(0)
            b64 = base64.b64encode(excel_file.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="feedback_trends.xlsx">Download Excel File</a>'
            st.markdown(href, unsafe_allow_html=True)

