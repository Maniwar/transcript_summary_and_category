import streamlit as st
import pandas as pd
import base64
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chardet
import io
import math
import numpy as np

# Initialize BERT model
@st.cache(allow_output_mutation=True)
def initialize_bert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Function to preprocess the text
def preprocess_text(text):
    # Convert to string if input is a real number
    if isinstance(text, float) and math.isfinite(text):
        text = str(text)

    # Remove unnecessary characters and weird characters
    text = text.encode('ascii', 'ignore').decode('utf-8')

    # Return the text without removing stop words
    return text.strip()

# Streamlit interface
st.title("👨‍💻 Chat Transcript Categorization")

# Add file uploader for the CSV
transcript_file = st.file_uploader("Upload CSV file", type="csv")

# Main layout
st.header("Processing")

# Only process if a file is uploaded
if transcript_file is not None:
    start_processing = st.button('Start Processing')

    # Read the uploaded CSV file with different encoding types
    raw_data = transcript_file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    try:
        df = pd.read_csv(io.BytesIO(raw_data), encoding=encoding)
    except UnicodeDecodeError:
        st.error("Error: Unable to decode the CSV file. Please try a different encoding type.")

    # Display a dropdown to select the transcript column
    selected_column = st.selectbox("Select transcript column", df.columns)

    # Initialize BERT model
    bert_model = initialize_bert_model()

    # Define extended categories and subcategories for customer call intents
    customer_categories = {
        "Product Related": [
            "Product Quality",
            "Product Availability",
            "Product Specifications",
            "Product Compatibility",
            "Product Warranties and Guarantees",
            "Product Installation",
            "Product Maintenance",
            "Refund Requests Due to Product Issues",
            "Promo Code Requests Due to Product Issues"
        ],
        "Service Queries": [
            "Service Types (Standard, Express, International)",
            "Service Rates",
            "Service Customization",
            "Service Experience",
            "Refund Requests Due to Service Issues",
            "Promo Code Requests Due to Service Issues"
        ],
        "Order Issues": [
            "Wrong Product Received",
            "Missing Items",
            "Order Tracking",
            "Delivery Delays",
            "Damaged Package",
            "Order Modifications",
            "Bulk Orders",
            "Custom Orders",
            "Return Requests",
            "Refund Requests Due to Order Issues",
            "Promo Code Requests Due to Order Issues"
        ],
        "Billing and Invoice Issues": [
            "Unauthorized Payments",
            "Billing Errors",
            "Invoice Discrepancy",
            "Refund Requests Due to Billing Issues",
            "Promo Code Requests Due to Billing Issues"
        ],
        "Payment Process": [
            "Payment Errors",
            "Credit/Debit Card",
            "Bank Account",
            "Digital Wallet",
            "Payment Plans",
            "Gift Cards",
            "Promo Code Requests Due to Payment Issues",
            "Refund Requests Due to Payment Issues"
        ],
        "Account Management": [
            "Login",
            "Password Reset",
            "Security",
            "Subscription/Membership",
            "Communication Preferences",
            "Personal Data Handling",
            "Account Deactivation",
            "Refund Requests Due to Account Management Issues",
            "Promo Code Requests Due to Account Management Issues"
        ],
        "Technical Issues": [
            "Website Navigation",
            "App Errors",
            "Payment Gateway",
            "Accessibility",
            "Computer or Laptop Technical Issues",
            "Browser Compatibility",
            "Security Warnings",
            "Promo Code Requests Due to Technical Issues",
            "Refund Requests Due to Technical Issues"
        ],
        "Feedback and Suggestions": [
            "Product",
            "Service",
            "Website/App",
            "Suggestions"
        ],
        "Price Match": [
            "Price Match Requests",
            "Refund Requests Due to Price Match",
            "Promo Code Requests Due to Price Match"
        ],
    }

    st.sidebar.header("Edit Customer Categories")
    # Edit Customer categories
    customer_categories_edited = {}
    category_embeddings = {}  # Store precomputed embeddings
    for category, subcategories in customer_categories.items():
        category_subcategories = st.sidebar.text_area(f"{category} Customer Subcategories", value="\n".join(subcategories))
        customer_categories_edited[category] = category_subcategories.split("\n")

        # Precompute embeddings for category keywords
        keyword_embeddings = [bert_model.encode(keyword) for keyword in customer_categories_edited[category]]
        category_embeddings[category] = np.array(keyword_embeddings)

    st.sidebar.subheader("Add or Modify Customer Categories")
    new_category_name = st.sidebar.text_input("New Customer Category Name")
    new_category_subcategories = st.sidebar.text_area(f"Subcategories for Customer Category {new_category_name}")
    if new_category_name and new_category_subcategories:
        customer_categories_edited[new_category_name] = new_category_subcategories.split("\n")
        # Precompute embeddings for new category keywords
        keyword_embeddings = [bert_model.encode(keyword) for keyword in customer_categories_edited[new_category_name]]
        category_embeddings[new_category_name] = np.array(keyword_embeddings)

    # Main processing
    if start_processing:
        # Create a progress bar in the main layout
        progress_bar = st.progress(0)
        progress_text = st.empty()

        # Calculate the number of steps to update the progress bar
        num_steps = len(df)
        step_size = 100 / num_steps

        # Initialize the progress
        progress = 0

        # Preprocess all lines outside the loop
        df[selected_column] = df[selected_column].apply(lambda x: preprocess_text(str(x)))

        # Process lines in batches
        batch_size = 100
        num_batches = math.ceil(len(df) / batch_size)
        for batch_index in range(num_batches):
            # Calculate the progress for the current batch
            progress = (batch_index + 1) / num_batches
            progress_bar.progress(progress)
            progress_text.text(f'Processing: {int(progress * 100)}%')

            # Get the lines for the current batch
            batch_start = batch_index * batch_size
            batch_end = (batch_index + 1) * batch_size
            batch_lines = df[selected_column].iloc[batch_start:batch_end].tolist()

            # Compute semantic similarity scores between customer comment and customer intents
            customer_intent_scores = {}
            customer_comment_embeddings = bert_model.encode(batch_lines)
            for intent, keyword_embeddings in category_embeddings.items():
                embedding_scores = cosine_similarity(customer_comment_embeddings, keyword_embeddings)
                customer_intent_scores[intent] = embedding_scores

            # Find the best matching customer category for each line in the batch
            best_customer_categories = np.argmax(customer_intent_scores, axis=0)
            best_customer_keywords = []
            best_customer_scores = []
            for i in range(len(best_customer_categories)):
                intent = best_customer_categories[i]
                intent_scores = customer_intent_scores[intent][i]
                keyword_index = np.argmax(intent_scores, default=0)
                best_customer_keyword = customer_categories_edited[intent][keyword_index]
                best_customer_keywords.append(best_customer_keyword)
                best_customer_scores.append(max(intent_scores, default=0))

            # Add the categorizations to the dataframe for the current batch
            df.at[batch_start:batch_end, "Best Matching Customer Category"] = best_customer_categories
            df.at[batch_start:batch_end, "Best Matching Customer Keyword"] = best_customer_keywords
            df.at[batch_start:batch_end, "Best Matching Customer Score"] = best_customer_scores

        # When all data is processed, set the progress bar to 100%
        progress_bar.progress(1.0)
        progress_text.text('Processing complete!')

        # Display the processed dataframe
        st.subheader("Processed Data")
        st.dataframe(df)

        # Generate a download link for the updated CSV file
        csv_data = df.to_csv(index=False, encoding='utf-8-sig')
        b64 = base64.b64encode(csv_data.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="processed_transcripts.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
