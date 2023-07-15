import pandas as pd
import nltk
from nltk.corpus import stopwords
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

# Set page title and layout
st.set_page_config(page_title="👨‍💻 Transcript Categorization")

# Initialize BERT model
@st.cache_resource
def initialize_bert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

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
    # Convert to string if input is a float
    if isinstance(text, float):
        text = str(text)

    # Remove unnecessary characters and weird characters
    text = text.encode('ascii', 'ignore').decode('utf-8')

    # Return the text without removing stop words
    return text

# Function to perform sentiment analysis
@st.cache_data
def perform_sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    return compound_score

# Function to summarize the text
@st.cache_resource
def summarize_text(text):
    summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarization_pipeline(text, max_length=400, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Function to compute semantic similarity
def compute_semantic_similarity(embedding1, embedding2):
    return (cosine_similarity([embedding1], [embedding2])[0][0] + 1) / 2

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
    ],
    "Stock & Availability": [
        "Product Was Out of Stock",
        "Had to Wait Too Long for Product Restock",
        "Product Was Excluded from Promotions",
        "Limited Availability for Deals",
        "Deal Restrictions Were Based on My Location",
        "Was Restricted on the Quantity I Could Purchase"
    ],
    "Pricing & Promotions": [
        "Promotion Didn't Apply to My Purchase",
        "Promotion Terms Were Unclear",
        "Saw Outdated Promotion Still Being Displayed",
        "Deal Fulfillment Didn't Go Through",
        "Had Trouble Locating Promotions on the Site",
        "Faced Issues When Applying Discounts",
        "Noticed Inconsistencies in Pricing",
        "Discounts Were Not Clearly Visible",
        "Encountered Problems with Membership Program",
        "Felt Pricing Was Deceptive",
        "Confused Over How Bulk Discounts Applied",
        "Lacked Information on Seasonal Sales",
        "Faced Problems with Referral Program",
        "Encountered Unexpected Fees"
    ],
    "Pre-Order & Delivery Planning": [
        "Encountered Problems During Pre-Order",
        "Experienced Delays in Pre-Order",
        "Received Inaccurate Pre-Order Information",
        "Pre-Order Was Cancelled Unexpectedly",
        "Faced Unexpected Charges for Pre-Order",
        "Didn't Receive Updates on Pre-Order Status",
        "Had Issues with Delivery of Pre-Ordered Product",
        "Pre-Order Process Was Confusing",
        "Couldn't Pre-Order the Product",
        "Delivery Timelines Were Unclear",
        "Mismatch Between Pre-Ordered and Delivered Product",
        "Had Issues with Partial Payments",
        "Had Trouble Modifying Pre-Order Details",
        "Limited Options for Delivery Date",
        "Had Issues with Delivering Split Orders"
    ],
    "Website & App Interface": [
        "Website Layout Was Confusing",
        "Had Trouble Finding Important Features",
        "Encountered Broken Links",
        "Couldn't Find Contact Information",
        "Faced Problems with the Search Function",
        "Didn't Understand the Navigation Menu",
        "Had Issues with Filtering and Sorting",
        "Website Speed Was Slow",
        "Received Errors When Loading Pages",
        "Website Wasn't Mobile Friendly",
        "Had Trouble Accessing Account Information",
        "Encountered Problems with Account Registration",
        "Forgot Password Functionality Wasn't Working",
        "Had Issues with Saving Items to Cart",
        "Encountered Problems with Checkout Process"
    ],
    "Technical Issues": [
        "Website Went Down Unexpectedly",
        "Encountered Errors When Trying to Complete a Task",
        "Slow Website Performance",
        "Couldn't Access Website/App",
        "Had Trouble Logging In",
        "Account Settings Were Not Saving",
        "Received Error Messages",
        "Had Trouble Completing a Purchase",
        "Website Was Unresponsive",
        "Encountered Problems with Mobile App",
        "Couldn't Upload Files",
        "Had Issues with Password Reset",
        "Encountered Problems with Notifications",
        "Had Trouble with Live Chat Support",
        "Couldn't Contact Customer Support"
    ]
}

keywords_text = {}
keyword_embeddings = {}

for category in default_categories:
    keywords_text[category] = st.sidebar.text_area(f"Keywords for {category} (One per line)", "\n".join(default_categories[category]))
    keywords_text[category] = [keyword.strip() for keyword in keywords_text[category].split("\n")]
    keyword_embeddings.update(compute_keyword_embeddings(keywords_text[category]))

# Convert keywords to lowercase
keyword_embeddings = {keyword.lower(): embedding for keyword, embedding in keyword_embeddings.items()}

# Get user input comment
comment = st.text_area("Enter your comment here")

# Preprocess the comment
comment = preprocess_text(comment)

# Encode the comment
model = initialize_bert_model()
comment_embedding = model.encode([comment])[0]

# Compute similarity scores and store them in a DataFrame
similarity_scores = pd.DataFrame(columns=['Keyword', 'Similarity Score'])

for main_category, keywords in keywords_text.items():
    for keyword in keywords:
        keyword_embedding = keyword_embeddings[keyword.lower()]
        similarity_score = compute_semantic_similarity(keyword_embedding, comment_embedding)
        similarity_scores = similarity_scores.append({'Keyword': keyword, 'Similarity Score': similarity_score}, ignore_index=True)

# Sort the DataFrame by similarity score
similarity_scores = similarity_scores.sort_values(by='Similarity Score', ascending=False)

# Print similarity scores
st.subheader("Similarity Scores")
st.dataframe(similarity_scores)
