ValueError: 10 columns passed, passed data had 9 columns
Traceback:
File "C:\Python311\Lib\site-packages\streamlit\runtime\scriptrunner\script_runner.py", line 552, in _run_script
    exec(code, module.__dict__)
File "C:\Users\m.berenji\Desktop\To Move\git\NPS Script\transcript_categories\transcript_category_summary_csv.py", line 349, in <module>
    trends_data = process_feedback_data(feedback_data, comment_column, date_column, categories, similarity_threshold, similarity_score, best_match_score)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Python311\Lib\site-packages\streamlit\runtime\caching\cache_utils.py", line 211, in wrapper
    return cached_func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Python311\Lib\site-packages\streamlit\runtime\caching\cache_utils.py", line 240, in __call__
    return self._get_or_create_cached_value(args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Python311\Lib\site-packages\streamlit\runtime\caching\cache_utils.py", line 266, in _get_or_create_cached_value
    return self._handle_cache_miss(cache, value_key, func_args, func_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Python311\Lib\site-packages\streamlit\runtime\caching\cache_utils.py", line 320, in _handle_cache_miss
    computed_value = self._info.func(*func_args, **func_kwargs)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\m.berenji\Desktop\To Move\git\NPS Script\transcript_categories\transcript_category_summary_csv.py", line 330, in process_feedback_data
    trends_data = pd.DataFrame(categorized_comments, columns=headers)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Python311\Lib\site-packages\pandas\core\frame.py", line 745, in __init__
    arrays, columns, index = nested_data_to_arrays(
                             ^^^^^^^^^^^^^^^^^^^^^^
File "C:\Python311\Lib\site-packages\pandas\core\internals\construction.py", line 510, in nested_data_to_arrays
    arrays, columns = to_arrays(data, columns, dtype=dtype)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Python311\Lib\site-packages\pandas\core\internals\construction.py", line 875, in to_arrays
    content, columns = _finalize_columns_and_data(arr, columns, dtype)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Python311\Lib\site-packages\pandas\core\internals\construction.py", line 972, in _finalize_columns_and_data
    raise ValueError(err) from err

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
from io import BytesIO
import streamlit as st

# Set page title and layout
st.set_page_config(page_title="👨‍💻 Feedback Categorization")

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
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Streamlit interface
st.title("👨‍💻 Feedback Categorization")

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
    "Product Discovery Issues": [
        "Difficulty Finding Products",
        "Insufficient Product Information",
        "Mobile Webpage Layout Problems",
        "Search Functionality Problems",
        "Issues with Product Filters",
        "User Interface Discrepancies"
    ],
    "Product Selection Issues": [
        "Problems Comparing Products",
        "Confusing Product Descriptions",
        "Issues with Configurator Performance",
        "Configurator Inaccuracies",
        "Usability Issues with Configurator",
        "Limited Product Varieties",
        "Unavailability of Certain Sizes",
        "Product Not Available In my Area or Zip code",
        "Product Availability Best Buy Pick-up Option Unclear ",
        "Frequent Stock Shortages"
    ],
    "Cart & Checkout Issues": [
        "Complexities in Cart Management",
        "Items were removed from my Cart",
        "Product Unavailability in Cart",
        "Limited Shipping Methods",
        "Complicated Checkout Process",
        "Inconveniences in Store Pick-up",
        "Delivery Date Calculation Errors",
        "Address Verification Problems"
    ],
    "Order Processing Issues": [
        "My order keeps getting cancelled",
        "Delays due to Stock Issues",
        "Order Modification Difficulties"
    ],
    "Payment Processing Issues": [
        "Payment Declined",
        "Unauthorized Charges",
        "Payment Gateway Errors",
        "Payment Verification Issues",
        "Refund Delays",
        "Coupon Code Malfunctions"
    ],
    "Delivery Issues": [
        "Promised Delivery Date was Missed or Late",
        "Order was not Delivered",
        "Problems with UPS",
        "Problems with AGS",
        "Problems with XPO",
        "Problems with FEDEX",
        "Poor Condition of Delivered Packages",
        "Problems with In-store Pick-up",
        "Package contained wrong product",
        "Missing Items in Delivered Package",
        "Received Damaged or Defective Item",
        "Limited Delivery Zones"
    ],
    "Installation Problems": [
        "Complex Installation Process",
        "Unprofessional Installation Process",
        "Vague Installation Instructions",
        "Missing Installation Parts",
        "Issues Detected Post-Installation",
        "Inadequate Installation Support",
        "Incompatibility of Products"
    ],
    "Service & Repair Problems": [
        "Inferior On-Site Service",
        "Delayed Service Response",
        "Substandard Repair Work",
        "Slow Repair Process",
        "Lack of Cost Transparency in Service & Repair",
        "Poor Communication during Service & Repair",
        "Insufficient Warranty Coverage"
    ],
    "Customer Support Issues": [
        "Long Wait Times for Support Response",
        "Unsatisfactory Support Quality",
        "Multiple Chat Transfers",
        "Limited Knowledge of Support Staff",
        "Unresolved Customer Issues",
        "Poor Follow-up Support",
        "Difficulty Reaching Support",
        "Unresponsive Support Staff",
        "Inaccessible Support Channels",
        "Lack of Empathy from Support Staff"
    ],
    "Return & Refund Issues": [
        "Complex Return Process",
        "Delayed Refund Process",
        "Unclear Return Policy",
        "Disagreement over Return Shipping Responsibility",
        "Brief Return Window",
        "Exchange Policy Problems",
        "Refund Amount Discrepancies"
    ],
    "Website & Mobile App Performance Issues": [
        "Slow Website Load Speed",
        "Mobile App Usability Issues",
        "Website Design Criticisms",
        "Challenges Navigating Website",
        "Distractions from Pop-ups",
        "Poor Website Scrolling Experience"
    ],
    "Customer Communication Issues": [
        "Excessive Email Notifications",
        "Irrelevant Email Content",
        "Issues Unsubscribing from Emails",
        "Poor Response to Customer Feedback",
        "Delays in Communication"
    ],
    "Privacy & Security Issues": [
        "Concerns about Data Privacy",
        "Concerns about Data Security",
        "Problems with Login",
        "Two-Factor Authentication Difficulties",
        "Poor Account Management",
        "Lack of Transparency in Data Usage",
        "Experiences of Scams & Phishing",
        "Payment Information Security Concerns"
    ],
    "Product Feedback": [
        "Absence of Charger",
        "Product Durability Concerns",
        "Issues with Firmware",
        "General Product Feedback",
        "Problems with Mobile Apps",
        "Problems with Shop Mobile App",
        "Software Malfunctions",
        "Hardware Problems",
        "Poor Product Quality",
        "Poor Product Performance"
    ],
    "Policy Feedback": [
        "Inability to Replace Items",
        "Concerns about Device Locking Policies",
        "Feedback on Trade-In Process",
        "Shipping Policy Critiques",
        "Return Policy Critiques"
    ],
    "Unexpected Pricing": [
        "Unexpected Price Changes in Cart",
        "Issues with Applying Discounts",
        "Employee Purchase Program Difficulties",
        "First Responder Program Difficulties",
        "Lack of Pricing Transparency",
        "Uncompetitive Pricing"
    ]
}
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
        def process_feedback_data(feedback_data, comment_column, date_column, categories, similarity_threshold, similarity_score, best_match_score):
        # Compute keyword embeddings
        keyword_embeddings = compute_keyword_embeddings([keyword for keywords in categories.values() for keyword in keywords])
        
        # Initialize lists for categorized_comments, sentiments, similarity scores, and summaries
        categorized_comments = []
        sentiments = []
        similarity_scores = []
        summarized_texts = []
        categories_list = []
    
        # Process each comment
        for index, row in feedback_data.iterrows():
            preprocessed_comment = preprocess_text(row[comment_column])
            summarized_text = summarize_text(preprocessed_comment)
            comment_embedding = initialize_bert_model().encode([summarized_text])[0]  # Compute the comment embedding once
            sentiment_score = perform_sentiment_analysis(preprocessed_comment)
            category = 'Other'
            sub_category = 'Other'
            best_match_score = float('-inf')  # Initialized to negative infinity
    
            # Tokenize the preprocessed_comment
            tokens = word_tokenize(preprocessed_comment)
    
            for main_category, keywords in categories.items():
                for keyword in keywords:
                    keyword_embedding = keyword_embeddings[keyword]  # Use the precomputed keyword embedding
                    similarity_score = compute_semantic_similarity(keyword_embedding, comment_embedding)
                    # If similarity_score equals best_match_score, we pick the first match.
                    # If similarity_score > best_match_score, we update best_match.
                    if similarity_score >= best_match_score:
                        category = main_category
                        sub_category = keyword
                        best_match_score = similarity_score
    
            # If in emerging issue mode and the best match score is below the threshold, set category and sub-category to 'No Match'
            if emerging_issue_mode and best_match_score < similarity_threshold:
                category = 'No Match'
                sub_category = 'No Match'
    
            parsed_date = row[date_column].split(' ')[0] if isinstance(row[date_column], str) else None
            row_extended = row.tolist() + [preprocessed_comment, summarized_text, category, sub_category, sentiment_score, best_match_score, parsed_date]
            categorized_comments.append(row_extended)
            sentiments.append(sentiment_score)
            similarity_scores.append(similarity_score)
            summarized_texts.append(summarized_text)
            categories_list.append(category)
    
        # Create a new DataFrame with extended columns
        existing_columns = feedback_data.columns.tolist()
        additional_columns = [comment_column, 'Preprocessed Comment', 'Summarized Text', 'Category', 'Sub-Category', 'Sentiment', 'Best Match Score', 'Parsed Date']
        num_additional_columns = len(additional_columns)
        headers = existing_columns + additional_columns[:num_additional_columns]
        trends_data = pd.DataFrame(categorized_comments, columns=headers)
        trends_data['Summarized Text'] = summarized_texts
        trends_data['Category'] = categories_list
        trends_data['Parsed Date'] = pd.to_datetime(trends_data['Parsed Date'], errors='coerce').dt.date
        
        # Rename duplicate column names
        trends_data = trends_data.loc[:, ~trends_data.columns.duplicated()]
        duplicate_columns = set([col for col in trends_data.columns if trends_data.columns.tolist().count(col) > 1])
        for column in duplicate_columns:
            column_indices = [i for i, col in enumerate(trends_data.columns) if col == column]
            for i, idx in enumerate(column_indices[1:], start=1):
                trends_data.columns.values[idx] = f"{column}_{i}"
    
        return trends_data



        # Process feedback data and cache the result
        trends_data = process_feedback_data(feedback_data, comment_column, date_column, categories, similarity_threshold, similarity_score, best_match_score)

        # Display trends and insights
        if trends_data is not None:
            st.title("Feedback Trends and Insights")
            st.dataframe(trends_data)

            # Display pivot table with counts for Category, Sub-Category, and Parsed Date
            st.subheader("All Categories Trends")

            # Convert 'Parsed Date' into datetime format if it's not
            trends_data['Parsed Date'] = pd.to_datetime(trends_data['Parsed Date'], errors='coerce')

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
            pivot1.columns = ['Average Sentiment', 'Survey Count']
            pivot1 = pivot1.sort_values('Survey Count', ascending=False)

            pivot2 = trends_data.groupby(['Category', 'Sub-Category'])['Sentiment'].agg(['mean', 'count'])
            pivot2.columns = ['Average Sentiment', 'Survey Count']
            pivot2 = pivot2.sort_values('Survey Count', ascending=False)

            # Reset index for pivot2
            pivot2_reset = pivot2.reset_index()


            # Set 'Sub-Category' as the index
            pivot2_reset.set_index('Sub-Category', inplace=True)

            # Create and display a bar chart for pivot1 with counts
            st.bar_chart(pivot1['Survey Count'])

            # Display pivot table with counts for Category
            st.subheader("Category vs Sentiment and Survey Count")
            st.dataframe(pivot1)

            # Create and display a bar chart for pivot2 with counts
            st.bar_chart(pivot2_reset['Survey Count'])

            # Display pivot table with counts for Sub-Category
            st.subheader("Sub-Category vs Sentiment and Survey Count")
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
                top_comments = filtered_data.nlargest(10, 'Parsed Date')[['Parsed Date', comment_column,'Sentiment', 'Best Match Score']]

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

        # Save DataFrame and pivot tables to Excel
        excel_file = BytesIO()
        with pd.ExcelWriter(excel_file, engine='xlsxwriter', mode='xlsx') as excel_writer:
            trends_data.to_excel(excel_writer, sheet_name='Feedback Trends and Insights', index=False)

            # Convert 'Parsed Date' column to datetime type
            trends_data['Parsed Date'] = pd.to_datetime(trends_data['Parsed Date'], errors='coerce')

            # Create a separate worksheet for each category and sub-category
            for category, subcategories in categories.items():
                for subcategory in subcategories:
                    filtered_data = trends_data[(trends_data['Category'] == category) & (trends_data['Sub-Category'] == subcategory)]
                    filtered_data.to_excel(excel_writer, sheet_name=f'{category} - {subcategory}', index=False)

            # Create a separate worksheet for pivot tables
            pivot1.to_excel(excel_writer, sheet_name='Category vs Sentiment and Survey Count')
            pivot2_reset.to_excel(excel_writer, sheet_name='Sub-Category vs Sentiment and Survey Count')
            pivot.to_excel(excel_writer, sheet_name='Feedback Trends by Date')

        excel_file.seek(0)
        excel_data = excel_file.getvalue()

        # Download Excel file
        st.download_button("Download Excel", data=excel_data, file_name="feedback_trends.xlsx", mime="application/vnd.ms-excel")
