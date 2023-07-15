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
# Function to summarize the text
@st.cache_resource
def summarize_text(text, max_length=400, min_length=30):
    # Check if the text is less than 400 words
    if len(word_tokenize(text)) < 250:
        return text

    # Initialize the summarization pipeline
    summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

    # Split the text into chunks of approximately 1024 words
    text_chunks = textwrap.wrap(text, width=1024)

    # Initialize an empty string to store the full summary
    full_summary = ""

    # For each chunk of text...
    for chunk in text_chunks:
        # Summarize the chunk and add the result to the full summary
        summary = summarization_pipeline(chunk, max_length=max_length, min_length=min_length, do_sample=False)
        full_summary += summary[0]['summary_text'] + " "

    # Return the full summary
    return full_summary.strip()


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
        "Experienced Glitches on Website",
        "Website Performance Was Slow",
        "Had Trouble Navigating the Interface",
        "Found Broken Links on Website",
        "Couldn't Locate Features on Website",
        "User Experience Was Inconsistent Across Different Devices",
        "Website Went Down Unexpectedly",
        "User Interface Was Confusing",
        "Had Issues with Mobile Functionality",
        "Website Didn't Adjust Well to My Location"
    ],
    "Order Management & Checkout": [
        "Experienced Glitches During Ordering",
        "Checkout Process Was Too Lengthy",
        "Payment Failed During Checkout",
        "Had Issues with Shopping Cart",
        "Couldn't Modify My Order",
        "Had Trouble with Gift Wrapping or Special Instructions",
        "Couldn't Cancel My Order",
        "Didn't Receive Order Confirmation",
        "My Order Was Cancelled Unexpectedly",
        "No Option for Express Checkout",
        "Had Problems Reviewing Order Before Checkout"
    ],
    "Payment & Billing": [
        "Was Charged Incorrectly",
        "Payment Was Declined During Checkout",
        "Was Confused About Applying Discount/Coupon",
        "Unexpected Conversion Rates Were Applied",
        "Payment Options Were Limited",
        "Had Difficulty Saving Payment Information",
        "Noticed Suspicious Charges",
        "Had Problems with Installment Payment",
        "Had Difficulty Splitting Payment",
        "Concerned About Data Security During Payment",
        "Had Problems with Tax Calculation"
    ],
    "Delivery & Shipping": [
        "Delivery Was Late",
        "Product Wasn't Delivered",
        "Received Wrong Product",
        "Product Was Damaged Upon Arrival",
        "Delivery Was Incomplete",
        "Had Problems Tracking My Delivery",
        "Had Issues with the Courier Service",
        "Delivery Options Were Limited",
        "Product Packaging Was Poor",
        "Had Difficulties with International Shipping",
        "Delivery Restrictions Were Based on My Zip Code",
        "Didn't Receive Updates on Delivery Status",
        "Limited Options for Same-Day/Next-Day Delivery",
        "Fragile Items Were Handled Poorly During Delivery",
        "Didn't Get Adequate Communication from Courier",
        "Delivery Address Was Incorrect"
    ],
    "Store & Pickup Services": [
        "Had Trouble Locating Stores",
        "Mismatch Between Chosen and Actual Pickup Location",
        "Had to Change Pickup Store",
        "Instructions for In-Store Pickup Were Unclear",
        "Received Poor Support In-Store",
        "Waited Too Long for Pickup",
        "Had Difficulty Scheduling Pickup Time",
        "Confused About Store Purchase Return Policy",
        "Had Difficulty Accessing Order History"
    ],
    "Product Installation & Setup": [
        "Had Difficulty During Product Installation",
        "Installation Instructions Were Missing",
        "Had Problems with Product After Installation",
        "Didn't Get Enough Assistance During Setup",
        "Received Incorrect Assembly Parts",
        "Had Issues with Product Compatibility",
        "Faced Unexpected Requirements During Setup",
        "Mismatch Between Product and Manual",
        "Lacked Technical Support",
        "Didn't Find Troubleshooting Guides Helpful",
        "Needed Professional Installation",
        "Needed Additional Tools Unexpectedly",
        "Had Problems After Setup Update"
    ],
    "Returns, Refunds & Exchanges": [
        "Had Difficulty Initiating a Return",
        "I want a Refund",
        "I want an Exchange",
        "Refund Wasn't Issued After Return",
        "Was Ineligible for Return",
        "Confused Over Restocking Fees",
        "Had Problems with Pickup for Return",
        "Refund Process Was Too Long",
        "Discrepancies in Partial Refunds",
        "Had Difficulty Tracking Returned Item",
        "Had Difficulties with International Return",
        "Product Was Damaged During Return Shipping",
        "Limited Options for Size/Color Exchanges",
        "Had Problems with Return Label"
    ],
    "Pre-Purchase Assistance": [
        "Had Trouble Getting Help Before Buying",
        "Couldn't Find Enough Product Information or Advice",
        "Customer Service Took Too Long to Respond",
        "Received Incorrect Information",
        "Support Wasn't Helpful with Promotions or Deals",
        "Had Trouble Scheduling Store Visits or Pickups",
        "Got Inconsistent Information from Different Agents"
    ],
    "Post-Purchase Assistance": [
        "Had Trouble Contacting Support After Purchase",
        "Post-Purchase Customer Service Response Was Slow",
        "Issues Were Not Resolved by Customer Service",
        "Didn't Get Enough Help Setting Up or Using Product",
        "Trouble Understanding Product Features Due to Poor Support",
        "Had Difficulty Getting Assistance During Product Installation",
        "Had Trouble Contacting Customer Service for Delivery-Related Issues",
        "Had Trouble Escalating Issues",
        "Self-Service Options Were Limited"
    ],
    "Technical/Product Support": [
        "Need Help With Purchased Product",
        "Had Trouble Receiving Technical Support",
        "Troubleshooting Advice from Tech Support Wasn't Helpful",
        "Technical Issues Weren't Resolved by Support",
        "Technical Instructions Were Difficult to Understand",
        "Tech Support Didn't Follow Up",
        "Received Incorrect Advice from Tech Support",
        "Automated Technical Support Was Problematic",
        "Technical Support Hours Were Limited"
    ],
    "Repair Services": [
        "Had Trouble Scheduling Repair",
        "Repair Work Was Unsatisfactory",
        "Repair Took Too Long",
        "Confused About Repair Charges",
        "No Follow-Up After Repair",
        "Received Incorrect Advice from Repair Service",
        "Repair Resolution Process Was Inefficient"
    ],
    "Account Management": [
        "Had Difficulty Logging In",
        "Couldn't Retrieve Lost Password",
        "Didn't Receive Account Verification Email",
        "Had Trouble Changing Account Information",
        "Had Problems with Account Security",
        "Experienced Glitches During Account Creation",
        "Not Able to Deactivate Account",
        "Had Issues with Privacy Settings",
        "Account Was Suspended or Banned Unexpectedly",
        "Had Difficulty Linking Multiple Accounts",
        "Didn't Get Email Notifications",
        "Had Issues with Subscription Management",
        "Couldn't Track Order History",
        "Received Unwanted Marketing Emails",
        "Trouble Setting Preferences in Account"
    ],
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
        def process_feedback_data(feedback_data, comment_column, date_column, categories, similarity_threshold):
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
                if len(preprocessed_comment.split()) > 250:
                    summarized_text = summarize_text(preprocessed_comment)
                else:
                    summarized_text = preprocessed_comment
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
