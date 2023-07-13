import streamlit as st
import pandas as pd
import base64
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chardet
import io
import math

# Initialize BERT model
@st.cache_resource
def initialize_bert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Function to preprocess the text
@st.cache_data
def preprocess_text(text):
    # Convert to string if input is a real number
    if isinstance(text, float) and math.isfinite(text):
        text = str(text)

    # Remove unnecessary characters and weird characters
    text = text.encode('ascii', 'ignore').decode('utf-8')

    # Return the text without removing stop words
    return text.strip()

# Function to compute semantic similarity
@st.cache_data
def compute_semantic_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]


# Streamlit interface
st.title("👨‍💻 Chat Transcript Categorization")

# Add file uploader for the CSV
transcript_file = st.file_uploader("Upload CSV file", type="csv")

# Main layout
st.header("Process Your File")

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
    for category, subcategories in customer_categories.items():
        category_subcategories = st.sidebar.text_area(f"{category} Customer Subcategories", value="\n".join(subcategories))
        customer_categories_edited[category] = category_subcategories.split("\n")

    st.sidebar.subheader("Add or Modify Customer Categories")
    new_category_name = st.sidebar.text_input("New Customer Category Name")
    new_category_subcategories = st.sidebar.text_area(f"Subcategories for Customer Category {new_category_name}")
    if new_category_name and new_category_subcategories:
        customer_categories_edited[new_category_name] = new_category_subcategories.split("\n")

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

        # Process each line separately
        for i, row in df.iterrows():
            # Calculate the progress
            progress = (i + 1) / num_steps

            # Update the progress bar
            progress_bar.progress(progress)
            progress_text.text(f'Processing: {int(progress * 100)}%')

            # Extract the transcript line from the selected column
            line = row[selected_column]

            # Preprocess the line
            line = preprocess_text(str(line))

            # Compute semantic similarity scores between customer comment and customer intents
            customer_intent_scores = {}
            customer_comment_embedding = bert_model.encode(line)
            for intent, keywords in customer_categories_edited.items():
                embedding_scores = [compute_semantic_similarity(customer_comment_embedding, bert_model.encode(keyword)) for keyword in keywords]
                customer_intent_scores[intent] = embedding_scores

            # Find the best matching customer category
            best_customer_category = max(customer_intent_scores, key=lambda x: max(customer_intent_scores[x]), default="")

            # Find the best matching customer keyword
            best_customer_category_scores = customer_intent_scores[best_customer_category]
            best_customer_category_index = best_customer_category_scores.index(max(best_customer_category_scores, default=0))
            best_customer_category_keyword = customer_categories_edited[best_customer_category][best_customer_category_index]

            # Add the categorizations to the dataframe
            df.at[i, "Best Matching Customer Category"] = best_customer_category
            df.at[i, "Best Matching Customer Keyword"] = best_customer_category_keyword
            df.at[i, "Best Matching Customer Score"] = max(best_customer_category_scores, default=0)

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
