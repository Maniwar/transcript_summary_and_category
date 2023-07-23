import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import numpy as np
import xlsxwriter
import chardet
from transformers import pipeline
import base64
from io import BytesIO
import streamlit as st
import textwrap

# Set page title and layout
st.set_page_config(page_title="👨‍💻 Transcript Categorization")

# Initialize BERT model
@st.cache_resource
def initialize_bert_model():
    return SentenceTransformer('paraphrase-MiniLM-L12-v2')

# Create a dictionary to store precomputed embeddings
@st.cache_resource
def compute_keyword_embeddings(keywords):
    model = initialize_bert_model()
    keyword_embeddings = {}
    for keyword in keywords:
        keyword_embeddings[keyword] = model.encode([keyword])[0]
    return keyword_embeddings

# Function to preprocess the text
@st.cache_data
def preprocess_text(text):
    if isinstance(text, float):
        text = str(text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    return text

# Function to perform sentiment analysis
@st.cache_data
def perform_sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    return compound_score

# New function to summarize a list of texts
@st.cache_resource
def summarize_texts(texts, max_length=100, min_length=50):
    summarization_pipeline = pipeline("summarization", model="knkarthick/MEETING_SUMMARY")
    summaries_list = []
    for text in texts:
        text_chunks = textwrap.wrap(text, width=2000)
        summaries = summarization_pipeline(text_chunks, max_length=max_length, min_length=min_length, do_sample=False)
        full_summary = " ".join([summary['summary_text'] for summary in summaries])
        summaries_list.append(full_summary.strip())
    return summaries_list

# Function to compute semantic similarity
def compute_semantic_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

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

# Edit categories and keywords
st.sidebar.header("Edit Categories")
default_categories = {
    "Product Discovery & Selection": [
        "Had Trouble Searching for a Specific Product",
        "Found Product Information Unclear or Incomplete",
        "Lacked Enough Information in Product Reviews",
        "Product Specifications Seemed Inaccurate",
        "Product Images Seemed Outdated",
        "Couldn't Determine if Product Was in Stock",
        "Struggled to Compare Different Products",
        "Wanted Product Wasn't Available",
        "Confused About Different Product Options",
        "Overwhelmed by Too Many Product Options",
        "Product Filters Didn't Help Narrow Down Choices",
        "Products Seemed Misclassified",
        "Product Recommendations Didn't Seem Relevant",
        "Had Trouble Saving Favorite Products",
        "Didn't Get Enough Information from Product Manufacturer"
    ],}
categories = {}
for category, keywords in default_categories.items():
    category_name = st.sidebar.text_input(f"{category} Category", value=category)
    category_keywords = st.sidebar.text_area(f"Keywords for {category}", value="\n".join(keywords))
    categories[category_name] = category_keywords.split("\n")

st.sidebar.subheader("Add or Modify Categories")
new_category_name = st.sidebar.text_input("New Category Name")
new_category_keywords = st.sidebar.text_area(f"Keywords for {new_category_name}")
if new_category_name and new_category_keywords:
    categories[new_category_name] = new_category_keywords.split("\n")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

# Select the column containing the comments
comment_column = None
date_column = None
trends_data = None

# Define an empty DataFrame for feedback_data
feedback_data = pd.DataFrame()

if uploaded_file is not None:
    # Read customer feedback from uploaded file
    csv_data = uploaded_file.read()

    # Detect the encoding of the CSV file
    result = chardet.detect(csv_data)
    encoding = result['encoding']
    try:
        feedback_data = pd.read_csv(BytesIO(csv_data), encoding=encoding)
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
    comment_column = st.selectbox("Select the column containing the comments", feedback_data.columns.tolist())
    date_column = st.selectbox("Select the column containing the dates", feedback_data.columns.tolist())
    grouping_option = st.radio("Select how to group the dates", ["Date", "Week", "Month", "Quarter"])
    process_button = st.button("Process Feedback")

    if comment_column is not None and date_column is not None and grouping_option is not None and process_button:
        # Check if the processed DataFrame is already cached
        @st.cache_data
        def process_feedback_data(feedback_data, comment_column, date_column, categories, similarity_threshold, emerging_issue_mode):
            # Compute keyword embeddings
            keyword_embeddings = compute_keyword_embeddings([keyword for keywords in categories.values() for keyword in keywords])

            # Initialize lists for categorized_comments, sentiments, similarity scores, and summaries
            categorized_comments = []
            sentiments = []
            similarity_scores = []
            summarized_texts = []
            original_texts = []
            categories_list = []

            # Initialize the BERT model once
            model = initialize_bert_model()

            # Process each comment
            for index, row in feedback_data.iterrows():
                preprocessed_comment = preprocess_text(row[comment_column])
                original_texts.append(preprocessed_comment if len(preprocessed_comment.split()) <= 100 else None)

            # Summarize all the long comments
            long_comments_indices = [i for i, text in enumerate(original_texts) if text is not None]
            long_comments_texts = [text for text in original_texts if text is not None]
            long_comments_summaries = summarize_texts(long_comments_texts)
            
            # Substitute long comments with their summaries
            for i, summary in zip(long_comments_indices, long_comments_summaries):
                original_texts[i] = summary
            
            # Calculate sentiment and categorize for each text
            for index, preprocessed_comment in enumerate(original_texts):
                comment_embedding = model.encode([preprocessed_comment])[0]
                sentiments.append(perform_sentiment_analysis(preprocessed_comment))
                category = "NO MATCH"
                sub_category = "NO MATCH"
                best_match_score = 0.0
                for category_name, keywords in categories.items():
                    for keyword in keywords:
                        similarity_score = compute_semantic_similarity(comment_embedding, keyword_embeddings[keyword])
                        if similarity_score > best_match_score:
                            best_match_score = similarity_score
                            category = category_name
                            sub_category = keyword
                if emerging_issue_mode and best_match_score < similarity_threshold:
                    category = "NO MATCH"
                    sub_category = "NO MATCH"
                categories_list.append((category, sub_category, best_match_score))
            return original_texts, summarized_texts, sentiments, categories_list

        original_texts, summarized_texts, sentiments, categories_list = process_feedback_data(feedback_data, comment_column, date_column, categories, similarity_threshold, emerging_issue_mode)

        # Extend the feedback DataFrame with the preprocessed comments, summaries, sentiments, and categories
        additional_columns = [comment_column, 'Summarized Text', 'Category', 'Sub-Category', 'Sentiment', 'Best Match Score']
        extended_data = feedback_data.copy()

        for original_text, summarized_text, sentiment, category_info in zip(original_texts, summarized_texts, sentiments, categories_list):
            row_extended = [original_text, summarized_text] + list(category_info) + [sentiment]
            extended_data = extended_data.append(pd.Series(row_extended, index=additional_columns), ignore_index=True)

        # Group the data
        if grouping_option == "Date":
            extended_data[date_column] = pd.to_datetime(extended_data[date_column])
            trends_data = extended_data.groupby([date_column, 'Category']).size().reset_index(name='Counts')
        elif grouping_option == "Week":
            extended_data[date_column] = pd.to_datetime(extended_data[date_column]).dt.to_period('W')
            trends_data = extended_data.groupby([date_column, 'Category']).size().reset_index(name='Counts')
        elif grouping_option == "Month":
            extended_data[date_column] = pd.to_datetime(extended_data[date_column]).dt.to_period('M')
            trends_data = extended_data.groupby([date_column, 'Category']).size().reset_index(name='Counts')
        elif grouping_option == "Quarter":
            extended_data[date_column] = pd.to_datetime(extended_data[date_column]).dt.to_period('Q')
            trends_data = extended_data.groupby([date_column, 'Category']).size().reset_index(name='Counts')

        # Download the processed data
        processed_csv = extended_data.to_csv(index=False)
        b64 = base64.b64encode(processed_csv.encode()).decode()  # some strings
        linko = f'<a href="data:file/csv;base64,{b64}" download="processed_feedback.csv">Download Processed Feedback</a>'
        st.markdown(linko, unsafe_allow_html=True)

        # Download the trend data
        trends_csv = trends_data.to_csv(index=False)
        b64 = base64.b64encode(trends_csv.encode()).decode()  # some strings
        linko = f'<a href="data:file/csv;base64,{b64}" download="trends_data.csv">Download Trend Data</a>'
        st.markdown(linko, unsafe_allow_html=True)
