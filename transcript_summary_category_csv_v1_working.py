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
combined_categories = {
    "Product Discovery": [
        "Product Information Misrepresentation",
        "Insufficient Product Description Detail",
        "Absence of Product Reviews",
        "Incorrect Product Specifications Display",
        "Obsolete Product Images",
        "Inadequate Product Availability Details",
        "Challenges in Comparing Similar Products",
        "Desired Products Unavailable",
        "Complications with Product Variants Selection",
        "Delivered Item Mismatch with Selected Product Variant",
        "Unclear Marking of Out of Stock Items",
        "Delay in Restocking Popular Variants",
        "Excessive Product Options",
        "Limited Product Selection Filters",
        "Incorrect Product Category Assignment",
        "Poor Quality Product Recommendations",
        "Challenges in Saving Preferred Products for Future Reference",
        "Absence of Details About Product Origin or Manufacturer"
    ],
    "Promotions & Deals": [
        "Failure in Applying Promotion Code",
        "Unclear Promotion Terms and Conditions",
        "Display of Expired Promotion",
        "Promised Deal not Honored",
        "Limited Availability of Promotional Deals",
        "Discrepancies with Membership Discounts",
        "Pricing Inconsistencies",
        "Inaccessible or Hidden Discounts",
        "Complications with Employee Purchasing Program",
        "Discrepancies in Applying Promotional Discounts",
        "Perception of Deceptive Pricing",
        "Confusion Regarding Bulk Discount Structure",
        "Insufficient Information on Seasonal Sales",
        "Exclusion of Specific Products from Promotions",
        "Complications with Referral Bonus System",
        "Geographical Limitations for Deals",
        "Discrepancies with Coupon/Discount Application"
    ],
    "Pre-order Concerns": [
        "Inability to Pre-order Desired Product",
        "Pre-order Delivery Delays",
        "Incorrect Pre-order Information Displayed",
        "Unanticipated Pre-order Cancellations",
        "Unjustified Charges for Pre-order",
        "Absence of Updates on Pre-order Status",
        "Delivery Complications with Pre-ordered Items",
        "Confusion in Pre-order Process",
        "Unavailable Pre-order Option for Desired Product",
        "Unclear Pre-order Delivery Timelines",
        "Mismatch Between Pre-ordered Item and Final Product",
        "Complications with Part-payment for Pre-orders",
        "Inability to Modify Pre-order Details"
    ],
    "Website/App Navigation": [
        "Website/App Functionality Errors",
        "Slow Website/App Performance",
        "Challenges in Interface Navigation",
        "Broken Website/App Links",
        "Difficulty Locating Specific Features",
        "Inconsistent User Experience Across Devices",
        "Unexpected Website/App Downtime",
        "Confusing User Interface Design",
        "Mobile Website/App Complications",
        "Challenges in Saving Payment and Shipping Information",
        "Wishlist Functionality Complications",
        "Absence of Live Chat Support",
        "Challenges in Accessing Order History",
        "Inadequate Search Engine Functionality",
        "Language Localization Complications"
    ],
    "Ordering & Checkout": [
        "Errors During Ordering Process",
        "Extended Checkout Process Duration",
        "Failure in Payment Processing",
        "Shopping Cart Functionality Complications",
        "Unexpected Additional Fees",
        "Challenges in Modifying Order Details",
        "Complications with Gift Wrapping or Special Instructions",
        "Complications in Order Cancellation",
        "Absence of Order Confirmation",
        "Unexpected Order Cancellation",
        "Unexplained Order Cancellation",
        "Restrictions on Limited Quantity Purchase",
        "Absence of Delivery Date Selection Option",
        "Inability to Split Order Deliveries",
        "Unavailable Express Checkout Option",
        "Challenges in Order Review Process"
    ],
    "Payment Processing": [
        "Discrepancies in Amount Charged",
        "Unsuccessful Refund Attempt",
        "Payment Declined by Bank",
        "Unexpected Currency Conversion Rates",
        "Limited Payment Options",
        "Complications in Storing Payment Information",
        "Suspicious Charges on Card",
        "Complications with Monthly Installment Payments",
        "Inability to Split Payments Across Multiple Methods",
        "Concerns About Stored Payment Details Security",
        "Errors in Tax Calculation"
    ],
    "Delivery & Shipping": [
        "Late Delivery of Order",
        "Non-Delivery of Product",
        "Wrong Product Delivered",
        "Product Damaged Upon Arrival",
        "Missing Items in Delivery",
        "Complications with Delivery Tracking",
        "Poor Service by Delivery Company",
        "Limited Delivery Options",
        "Poor Packaging Quality",
        "Challenges with International Shipping",
        "Unavailable Products for Certain ZIP Codes",
        "Inadequate Delivery Updates",
        "Absence of Same-Day/Next-Day Delivery Options",
        "Poor Handling of Fragile Items",
        "Unsatisfactory Communication from Courier",
        "Incorrect Delivery Address Recorded"
    ],
    "Pickup & Store Location": [
        "Difficulty Locating Pickup Store",
        "Mismatch Between Selected Store and Available Pickup Locations",
        "Post-Order Pickup Store Changes",
        "Unclear Instructions for In-Store Pickup",
        "Inadequate In-store Customer Support",
        "Extended Wait Time for Pickup",
        "Challenges in Scheduling Pickup Time",
        "Unclear Return Policy for Store-Purchased Items"
    ],
    "Installation & Setup": [
        "Challenges in Product Installation",
        "Missing Installation Instructions",
        "Non-Functional Product Post-Installation",
        "Unavailable Setup Assistance",
        "Incorrect Assembly Parts Included",
        "Product Compatibility Complications",
        "Unexpected Additional Setup Requirements",
        "Mismatch Between Product and Manual",
        "Absence of Technical Support",
        "Inadequate Troubleshooting Guides",
        "Requirement for Professional Installation",
        "Unexpected Need for Additional Tools for Installation",
        "Complications with Product Updates Post-Setup"
    ],
    "Returns & Refunds": [
        "Challenges in Initiating Return Process",
        "Refund Not Issued as Promised",
        "Ineligibility for Product Return",
        "Restocking Fee Complications",
        "Challenges with Return Pickup Arrangement",
        "Extended Wait for Refund Approval",
        "Partial Refund Discrepancies",
        "Challenges in Tracking Return Status",
        "Complications with Product Exchange",
        "Challenges with International Returns",
        "Damage Incurred During Return Shipping",
        "Absence of Exchange for Different Sizes/Colors",
        "Complications with Return Labels"
    ],
    "Customer Service & Support": [
        "Inadequate Customer Service Interaction",
        "Queries Not Fully Resolved",
        "Extended Response Times from Customer Service",
        "Unavailable Customer Support",
        "Customer Support Misinformation",
        "Absence of Follow-up from Customer Service",
        "Difficulties in Escalating Issues",
        "Language Barriers in Customer Service",
        "Inefficient Complaint Resolution",
        "Limited Self-Service Support Options",
        "Incorrect Advice from Customer Support",
        "Automated Response System Complications",
        "Limited Customer Service Hours",
        "Absence of Multilingual Support",
        "Non-responsive Customer Support Email",
        "Poor Experience with Phone Support"
    ],
    "Product Quality": [
        "Inferior Product Material Quality",
        "Short Product Lifespan",
        "Product Not Functioning as Expected",
        "Product Safety Concerns",
        "Missing Parts in Delivered Product",
        "Inadequate Product Packaging",
        "Discrepancy Between Product Image and Actual Product",
        "Unexpected Product Size/Dimensions",
        "Product Performance Below Expectations",
        "Undisclosed Product Faults",
        "Product Quality Inconsistent with Price",
        "Challenges with Product Maintenance/Cleaning",
        "Poor Performance of Electrical/Electronic Components",
        "Unintuitive Product Design",
        "Product Not Suitable for Intended Use"
    ],
    "Subscription & Membership": [
        "Difficulty in Subscription Activation",
        "Unjustified Subscription Renewal",
        "Challenges in Subscription Cancellation",
        "Unexpected Subscription Fees",
        "Limited Benefits in Membership Program",
        "Inability to Opt Out from Recurring Subscription",
        "Subscription Not Activated as Promised",
        "Complications with Membership Tier Benefits",
        "Absence of Desired Membership Options",
        "Inadequate Communication on Subscription Changes",
        "Subscription Service Below Expectations",
        "Limited Subscription Payment Options",
        "Membership Renewal Complications",
        "Absence of Family/Group Membership Plans"
    ],
    "Account Information": [
        "Inability to Delete Account",
        "Unexpected Account Suspension",
        "Inability to Create Account",
        "Limited Account Recovery Options",
        "Complications in Managing Privacy Settings",
        "Unwanted Recommendation Based on Browsing History",
        "Absence of Two-Factor Authentication",
        "Inadequate Control Over Personal Data",
        "Challenges in Account Login",
        "Inability to Update Account Information",
        "Unintended Account Deactivation",
        "Difficulty in Password Reset",
        "Absence of Social Media Login Options",
        "Complications with Account Verification",
        "Unrecognized Devices/Activities on Account",
        "Unauthorized Account Access"
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
