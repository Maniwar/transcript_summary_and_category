import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Initialize BERT model
@st.cache_resource  # Cache the BERT model as a resource
def initialize_bert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Initialize T5 model and tokenizer
@st.cache_resource  # Cache the T5 model and tokenizer as resources
def initialize_t5_model():
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    return model, tokenizer

# Function to preprocess the text
def preprocess_text(text):
    # Convert to string if input is a float
    if isinstance(text, float):
        text = str(text)

    # Remove unnecessary characters and weird characters
    text = text.encode('ascii', 'ignore').decode('utf-8')

    # Return the text without removing stop words
    return text

# Function for ML summarization
def ml_summarize(text, model, tokenizer):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=150, min_length=40, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Function to compute semantic similarity
def compute_semantic_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]


# Streamlit interface
st.title("👨‍💻 Chat Transcript Categorization")

# Add text input for the chat transcript
transcript_text = st.text_area("Paste the chat transcript here")

# Add a button to trigger the processing
process_button = st.button("Process")

# Only process if the button is clicked
if process_button:
    # Preprocess the chat transcript
    transcript_lines = transcript_text.split("\n")
    transcript_lines = [line.strip() for line in transcript_lines if line.strip()]

    # Split the transcript into agent and customer comments
    agent_comments = []
    customer_comments = []
    for line in transcript_lines:
        if line.startswith("Agent:"):
            agent_comments.append(line[6:].strip())
        elif line.startswith("Customer:"):
            customer_comments.append(line[9:].strip())

    # Preprocess agent and customer comments
    agent_comments = [preprocess_text(comment) for comment in agent_comments]
    customer_comments = [preprocess_text(comment) for comment in customer_comments]

    # Concatenate agent and customer comments
    agent_text = ' '.join(agent_comments)
    customer_text = ' '.join(customer_comments)

    # Initialize BERT model and T5 model
    bert_model = initialize_bert_model()
    t5_model, t5_tokenizer = initialize_t5_model()

    # ML summarization for agent and customer parts
    agent_summary = ml_summarize(agent_text, t5_model, t5_tokenizer)
    customer_summary = ml_summarize(customer_text, t5_model, t5_tokenizer)

    st.subheader("Customer Summary:")
    st.write(customer_summary)

    # Display the agent and customer summaries
    st.subheader("Agent Summary:")
    st.write(agent_summary)

    # Define extended categories and subcategories for customer call intents
    customer_categories = {
        "Product Queries": [
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
            "Browser Compatibility",
            "Security Warnings",
            "Promo Code Requests Due to Technical Issues",
            "Refund Requests Due to Technical Issues"
        ],
        "Feedback and Suggestions": [
            "Product",
            "Service",
            "Website/App",
            "Suggestions",
            "Promo Code Requests for Feedback",
            "Refund Requests for Feedback"
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
        "Order Management": [
            "Order Status Updates",
            "Handling Returns",
            "Processing Order Cancellations",
            "Organizing Product Exchanges",
            "Assisting with Order Modifications",
            "Helping with Bulk Orders",
            "Customizing Orders",
        ],
        "Billing and Invoice Assistance": [
            "Handling Unauthorized Payments",
            "Resolving Billing Errors",
            "Explaining Invoices",
            "Processing Refunds",
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

    # Initialize BERT model
    model = initialize_bert_model()

    # Compute embeddings for customer categories and subcategories
    customer_embeddings = {}
    for category, subcategories in customer_categories_edited.items():
        for subcategory in subcategories:
            embedding = model.encode(subcategory)
            customer_embeddings[category + ' - ' + subcategory] = embedding

    # Compute semantic similarity scores between agent summary and customer intents
    intent_scores = {}
    agent_summary_embedding = model.encode(agent_summary)
    for intent, embedding in customer_embeddings.items():
        score = compute_semantic_similarity(agent_summary_embedding, embedding)
        intent_scores[intent] = score

    # Find the best matching intent
    best_intent = max(intent_scores, key=intent_scores.get)
    best_intent_score = intent_scores[best_intent]

    # Compute embeddings for agent actions
    agent_embeddings = {}
    for category, subcategories in agent_categories_edited.items():
        for subcategory in subcategories:
            embedding = model.encode(subcategory)
            agent_embeddings[category + ' - ' + subcategory] = embedding

    # Compute semantic similarity scores between customer summary and agent actions
    action_scores = {}
    customer_summary_embedding = model.encode(customer_summary)
    for action, embedding in agent_embeddings.items():
        score = compute_semantic_similarity(customer_summary_embedding, embedding)
        action_scores[action] = score

    # Find the best matching action
    best_action = max(action_scores, key=action_scores.get)
    best_action_score = action_scores[best_action]

    # Display the best matching intent and action
    st.subheader("Best Matching Customer Intent:")
    st.write("Intent:", best_intent)
    st.write("Similarity Score:", best_intent_score)

    st.subheader("Best Matching Agent Action:")
    st.write("Action:", best_action)
    st.write("Similarity Score:", best_action_score)
