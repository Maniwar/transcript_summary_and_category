IndexError: list index out of range
Traceback:
File "C:\Python311\Lib\site-packages\streamlit\runtime\scriptrunner\script_runner.py", line 552, in _run_script
    exec(code, module.__dict__)
File "C:\Users\m.berenji\Desktop\To Move\git\NPS Script\transcript_categories\batch_summary_transcript_v1broken.py", line 268, in <module>
    trends_data = process_feedback_data(feedback_data, comment_column, date_column, categories, similarity_threshold)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
File "C:\Users\m.berenji\Desktop\To Move\git\NPS Script\transcript_categories\batch_summary_transcript_v1broken.py", line 194, in process_feedback_data
    feedback_data['summarized_comments'] = feedback_data['preprocessed_comments'].apply(lambda x: summarize_text(x) if len(x.split()) > 100 else x)
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Python311\Lib\site-packages\pandas\core\series.py", line 4771, in apply
    return SeriesApply(self, func, convert_dtype, args, kwargs).apply()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Python311\Lib\site-packages\pandas\core\apply.py", line 1105, in apply
    return self.apply_standard()
           ^^^^^^^^^^^^^^^^^^^^^
File "C:\Python311\Lib\site-packages\pandas\core\apply.py", line 1156, in apply_standard
    mapped = lib.map_infer(
             ^^^^^^^^^^^^^^
File "pandas\_libs\lib.pyx", line 2918, in pandas._libs.lib.map_infer
File "C:\Users\m.berenji\Desktop\To Move\git\NPS Script\transcript_categories\batch_summary_transcript_v1broken.py", line 194, in <lambda>
    feedback_data['summarized_comments'] = feedback_data['preprocessed_comments'].apply(lambda x: summarize_text(x) if len(x.split()) > 100 else x)
                                                                                                  ^^^^^^^^^^^^^^^^^
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
File "C:\Users\m.berenji\Desktop\To Move\git\NPS Script\transcript_categories\batch_summary_transcript_v1broken.py", line 92, in summarize_text
    chunks = tokenize_and_chunk_text([text])
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
File "C:\Users\m.berenji\Desktop\To Move\git\NPS Script\transcript_categories\batch_summary_transcript_v1broken.py", line 83, in tokenize_and_chunk_text
    chunks[-1].extend(tokens)
    ~~~~~~^^^^
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
from transformers import pipeline, AutoTokenizer
import base64
from io import BytesIO
import streamlit as st
import textwrap
from categories_josh1 import default_categories
import time


# Set page title and layout
st.set_page_config(page_title="👨‍💻 Transcript Categorization")

# Initialize progress bar
progress = st.sidebar.progress(0)
progress_status = st.sidebar.empty()

# Initialize BERT model
@st.cache_resource
def initialize_bert_model():
    #return SentenceTransformer('all-MiniLM-L6-v2')
    #return SentenceTransformer('all-MiniLM-L12-v2')
    #return SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return SentenceTransformer('paraphrase-MiniLM-L12-v2')
    #return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    #return SentenceTransformer('stsb-roberta-base')
    #return SentenceTransformer('distilroberta-base-paraphrase-v1')

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

# Load the tokenizer corresponding to the model
st.cache_resource
tokenizer = AutoTokenizer.from_pretrained('knkarthick/MEETING_SUMMARY')

# Function to tokenize the text into chunks of approximately 1024 tokens each
@st.cache_resource
def tokenize_and_chunk_text(texts, max_tokens=1024):
    chunks = []
    num_tokens = 0
    for text in texts:
        tokens = tokenizer(text)['input_ids']
        if num_tokens + len(tokens) > max_tokens:
            chunks.append(tokens)
            num_tokens = len(tokens)
        else:
            chunks[-1].extend(tokens)
            num_tokens += len(tokens)
    return chunks


# Function to summarize the text
@st.cache_resource
def summarize_text(text, max_length=100, min_length=50):
    summarization_pipeline = pipeline("summarization", model="knkarthick/MEETING_SUMMARY")
    chunks = tokenize_and_chunk_text([text])
    if not chunks:  # If chunks is empty, add a new chunk with tokens
        chunks.append(tokens)
    else:
        chunks[-1].extend(tokens)
    summaries = []
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks, start=1):
        chunk_text = tokenizer.decode(chunk)
        summaries.append(summarization_pipeline(chunk_text, max_length=max_length, min_length=min_length, do_sample=False))
        progress.progress(i / total_chunks)
        progress_status.text(f"Summarizing text: {i}/{total_chunks}")
    full_summary = " ".join([summary['summary_text'] for summary in summaries])
    return full_summary.strip(), total_chunks




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
        def process_feedback_data(feedback_data, comment_column, date_column, categories, similarity_threshold):
            # Start timing
            start_time = time.time()
        
            # Compute keyword embeddings
            progress_status.text("Computing keyword embeddings...")
            keyword_embeddings = compute_keyword_embeddings([keyword for keywords in categories.values() for keyword in keywords])
        
            # Initialize lists for categorized_comments, sentiments, similarity scores, and summaries
            categorized_comments = []
            sentiments = []
            similarity_scores = []
            summarized_texts = []
            categories_list = []
        
            # Initialize the BERT model once
            model = initialize_bert_model()
        
            # Preprocess comments and summarize if necessary
            feedback_data['preprocessed_comments'] = feedback_data[comment_column].apply(preprocess_text)
            feedback_data['summarized_comments'] = feedback_data['preprocessed_comments'].apply(lambda x: summarize_text(x) if len(x.split()) > 100 else x)
        
            # Compute comment embeddings in chunks based on token count
            comment_texts = feedback_data['summarized_comments'].tolist()
            chunks = tokenize_and_chunk_text(comment_texts)
            comment_embeddings = []
            for chunk, _ in chunks:
                comment_embeddings.extend(model.encode(chunk))
            feedback_data['comment_embeddings'] = comment_embeddings
        
            # Compute sentiment scores
            feedback_data['sentiment_scores'] = feedback_data['preprocessed_comments'].apply(perform_sentiment_analysis)
        
            # Compute semantic similarity and assign categories in chunks
            for chunk, _ in chunks:
                batch_embeddings = model.encode(chunk)
                for main_category, keywords in categories.items():
                    for keyword in keywords:
                        keyword_embedding = keyword_embeddings[keyword]  # Use the precomputed keyword embedding
                        batch_similarity_scores = cosine_similarity([keyword_embedding], batch_embeddings)[0]
                        # Update categories and sub-categories based on the highest similarity score
                        for j, similarity_score in enumerate(batch_similarity_scores):
                            if j < len(categories_list):
                                if similarity_score > similarity_scores[j]:
                                    categories_list[j] = main_category
                                    summarized_texts[j] = keyword
                                    similarity_scores[j] = similarity_score
                            else:
                                categories_list.append(main_category)
                                summarized_texts.append(keyword)
                                similarity_scores.append(similarity_score)
            # Prepare final data
            for index, row in feedback_data.iterrows():
                preprocessed_comment = row['preprocessed_comments']
                sentiment_score = row['sentiment_scores']
                category = categories_list[index]
                sub_category = summarized_texts[index]
                best_match_score = similarity_scores[index]
                summarized_text = row['summarized_comments']

                # If in emerging issue mode and the best match score is below the threshold, set category and sub-category to 'No Match'
                if emerging_issue_mode and best_match_score < similarity_threshold:
                    category = 'No Match'
                    sub_category = 'No Match'

                parsed_date = row[date_column].split(' ')[0] if isinstance(row[date_column], str) else None
                row_extended = row.tolist() + [preprocessed_comment, summarized_text, category, sub_category, sentiment_score, best_match_score, parsed_date]
                categorized_comments.append(row_extended)

            # Create a new DataFrame with extended columns
            existing_columns = feedback_data.columns.tolist()
            additional_columns = [comment_column, 'Summarized Text', 'Category', 'Sub-Category', 'Sentiment', 'Best Match Score', 'Parsed Date']
            headers = existing_columns + additional_columns
            trends_data = pd.DataFrame(categorized_comments, columns=headers)
            trends_data['Parsed Date'] = pd.to_datetime(trends_data['Parsed Date'], errors='coerce').dt.date

            # Rename duplicate column names
            trends_data = trends_data.loc[:, ~trends_data.columns.duplicated()]
            duplicate_columns = set([col for col in trends_data.columns if trends_data.columns.tolist().count(col) > 1])
            for column in duplicate_columns:
                column_indices = [i for i, col in enumerate(trends_data.columns) if col == column]
                for i, idx in enumerate(column_indices[1:], start=1):
                    trends_data.columns.values[idx] = f"{column}_{i}"

            # End timing
            end_time = time.time()
        
            progress_status.text(f"Processing complete. Total time: {end_time - start_time} seconds.")

            return trends_data



        # Process feedback data and cache the result
        trends_data = process_feedback_data(feedback_data, comment_column, date_column, categories, similarity_threshold)

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

            # Format column headers as date strings in 'YYYY-MM-DD' format
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
