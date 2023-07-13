import streamlit as st
import pandas as pd
import base64
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5ForConditionalGeneration, T5Tokenizer
import chardet
import io
import math

# Initialize BERT model
@st.cache_resource # Cache the BERT model as a resource
def initialize_bert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Initialize T5 model and tokenizer
@st.cache_resource # Cache the T5 model and tokenizer as resources
def initialize_t5_model():
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    return model, tokenizer

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

# Function for ML summarization
@st.cache_resource # Cache the ML summarization function as a resource
def ml_summarize(text, _model, _tokenizer):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=150, min_length=40, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Function to compute semantic similarity
@st.cache_data
def compute_semantic_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]


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

    # Initialize BERT model and T5 model
    bert_model = initialize_bert_model()
    t5_model, t5_tokenizer = initialize_t5_model()

    # Create empty lists to store summaries
    agent_summaries = []
    customer_summaries = []

    # Process each line separately
    for line in df[selected_column]:
        # Preprocess the line
        line = preprocess_text(str(line))

        # Split the line into agent and customer comments
        if line.startswith("Agent:"):
            agent_comment = line[6:].strip()
            agent_summaries.append(ml_summarize(agent_comment, t5_model, t5_tokenizer))
        elif line.startswith("Customer:"):
            customer_comment = line[9:].strip()
            customer_summaries.append(ml_summarize(customer_comment, t5_model, t5_tokenizer))

    # Join the agent and customer summaries
    agent_summary = ' '.join(agent_summaries)
    customer_summary = ' '.join(customer_summaries)

    # Display the agent and customer summaries
    st.subheader("Customer Summary:")
    st.write(customer_summary)

    st.subheader("Agent Summary:")
    st.write(agent_summary)
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

    # Define extended categories and subcategories for agent actions
    agent_categories = {
        "Product Assistance": [
            "Product Information",
            "Product Availability",
            "Product Specifications",
            "Product Alternatives",
            "Product Warranties and Guarantees",
            "Product Installation",
            "Product Maintenance",
        ],
        "Service Assistance": [
            "Explaining Service Types",
            "Providing Service Rates",
            "Service Customization",
            "Improving Service Experience"
        ],
        "Order Support": [
            "Order Status Updates",
            "Processing a Return related Refund",
            "Processing Order Cancellations",
            "Organizing Product Exchanges",
            "Assisting with Order Modifications",
            "Helping with Bulk Orders",
            "Customizing Orders",
            "Handling Unauthorized Payments",
            "Resolving Billing Errors",
            "Explaining Invoices",
            "Processing Non-return related Refund",
            "Issuing Promo Codes"
        ],
        "Payment Assistance": [
            "Resolving Payment Errors",
            "Assisting with Credit/Debit Card Issues",
            "Addressing Bank Account Concerns",
            "Providing Digital Wallet Support",
            "Explaining Payment Plans",
            "Helping with Gift Cards"
        ],
        "Account Maintenance": [
            "Resolving Login Issues",
            "Assisting with Password Reset",
            "Enhancing Security",
            "Managing Subscription and Membership",
            "Updating Communication Preferences",
            "Addressing Personal Data Queries",
            "Helping with Account Deactivation",
        ],
        "Technical Support": [
            "Supporting Website Navigation",
            "Troubleshooting App Errors",
            "Troubleshooting Computer or Laptop Technical Issues",
            "Resolving Payment Gateway Issues",
            "Improving Accessibility",
            "Fixing Browser Compatibility Issues",
            "Addressing Security Warnings",
        ],
        "Feedback Management": [
            "Collecting Product Feedback",
            "Collecting Service Feedback",
            "Collecting Website/App Feedback",
            "Accepting Suggestions",
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

    # Edit Agent categories
    st.sidebar.header("Edit Agent Categories")
    agent_categories_edited = {}
    for category, subcategories in agent_categories.items():
        category_subcategories = st.sidebar.text_area(f"{category} Agent Subcategories", value="\n".join(subcategories))
        agent_categories_edited[category] = category_subcategories.split("\n")

    st.sidebar.subheader("Add or Modify Agent Categories")
    new_category_name = st.sidebar.text_input("New Agent Category Name")
    new_category_subcategories = st.sidebar.text_area(f"Subcategories for Agent Category {new_category_name}")
    if new_category_name and new_category_subcategories:
        agent_categories_edited[new_category_name] = new_category_subcategories.split("\n")

    # Main processing
    if start_processing:
        # Create a progress bar in the main layout
        progress_bar = st.progress(0)
        progress_text = st.empty()

        # Calculate the number of steps to update the progress bar
        num_steps = len(df)
        step_size = 1 / num_steps

        # Initialize the progress
        progress = 0

        # Process each line separately
        for i, row in df.iterrows():
            # Update the progress bar
            progress += step_size
            progress_bar.progress(progress)
            progress_text.text(f'Processing: {int(progress * 100)}%')

            # Extract the transcript line from the selected column
            line = row[selected_column]

            # Preprocess the line
            line = preprocess_text(str(line))

            # Split the line into agent and customer comments
            if line.startswith("Agent:"):
                agent_comment = line[6:].strip()
                agent_summary = ml_summarize(agent_comment, t5_model, t5_tokenizer)
            elif line.startswith("Customer:"):
                customer_comment = line[9:].strip()
                customer_summary = ml_summarize(customer_comment, t5_model, t5_tokenizer)

            # Compute semantic similarity scores between agent summary and customer intents
            intent_scores = {}
            agent_summary_embedding = bert_model.encode(agent_summary)
            for intent, embeddings in customer_categories_edited.items():
                embedding_scores = []
                for embedding in embeddings:
                    embedding_scores.append(compute_semantic_similarity(agent_summary_embedding, bert_model.encode(embedding)))
                intent_scores[intent] = max(embedding_scores)

            # Find the best matching intent
            best_intent = max(intent_scores, key=intent_scores.get)
            best_intent_score = intent_scores[best_intent]

            # Compute semantic similarity scores between customer summary and agent actions
            action_scores = {}
            customer_summary_embedding = bert_model.encode(customer_summary)
            for action, embeddings in agent_categories_edited.items():
                embedding_scores = []
                for embedding in embeddings:
                    embedding_scores.append(compute_semantic_similarity(customer_summary_embedding, bert_model.encode(embedding)))
                action_scores[action] = max(embedding_scores)

            # Find the best matching action
            best_action = max(action_scores, key=action_scores.get)
            best_action_score = action_scores[best_action]

            # Add the summaries and categorizations to the dataframe
            df.at[i, "Agent Summary"] = agent_summary
            df.at[i, "Customer Summary"] = customer_summary
            df.at[i, "Best Matching Customer Intent"] = best_intent
            df.at[i, "Best Matching Agent Action"] = best_action

        # When all data is processed, set the progress bar to 100%
        progress_bar.progress(1.0)
        progress_text.text('Processing complete!')

        # Generate a download link for the updated CSV file
        csv_data = df.to_csv(index=False, encoding='utf-8-sig')
        b64 = base64.b64encode(csv_data.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="processed_transcripts.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

NameError: name 'tokenizer' is not defined
Traceback:
File "C:\Python311\Lib\site-packages\streamlit\runtime\scriptrunner\script_runner.py", line 552, in _run_script
    exec(code, module.__dict__)
File "C:\Users\m.berenji\Desktop\To Move\git\NPS Script\categorizer\transcript_category_csv.py", line 94, in <module>
    customer_summaries.append(ml_summarize(customer_comment, t5_model, t5_tokenizer))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
File "C:\Users\m.berenji\Desktop\To Move\git\NPS Script\categorizer\transcript_category_csv.py", line 39, in ml_summarize
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
             ^^^^^^^^^
