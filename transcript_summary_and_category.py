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


# Customer: Refund Request
with st.expander("Example script refund for product return"):
    transcript = """
    <div class="transcript">
    <b>Customer:</b> Hi, I recently purchased a product from your website, but I'm not satisfied with it at all. The quality is far below my expectations, and it doesn't function properly. I would like to request a refund. <br>
    <b>Agent:</b> I apologize for the inconvenience caused by the product. I understand your disappointment, and I'm here to assist you with the refund process. To proceed with the refund, I'll need some information from you. Could you please provide me with your order number and the reason for the refund? <br>
    <b>Customer:</b> Thank you for your understanding. My order number is #987654321, and the reason for the refund is the poor quality and malfunctioning of the product. It doesn't meet the specifications mentioned on your website. <br>
    <b>Agent:</b> I appreciate you providing the necessary details. I'll initiate the refund process immediately. Please note that it may take a few business days for the refund to be processed and reflect in your account. Is there anything else you would like to add regarding the refund? <br>
    <b>Customer:</b> No, that's all. I just want to make sure I receive the refund as soon as possible. This experience has been disappointing, and I hope the refund process is hassle-free. <br>
    <b>Agent:</b> I completely understand your concerns, and I assure you that we'll make the refund process as smooth as possible. Once the refund is processed, you'll receive an email notification confirming the transaction. If you have any further questions or need assistance during the process, please don't hesitate to reach out to us. <br>
    <b>Customer:</b> Thank you for your assurance. I'll be eagerly waiting for the refund confirmation email. I hope this issue gets resolved without any further complications. <br>
    <b>Agent:</b> You're welcome. We'll do our best to ensure a prompt resolution for you. Rest assured, I'll personally monitor the refund process and keep you informed of any updates. If you have any additional concerns, feel free to contact us. We appreciate your patience and cooperation. <br>
    <b>Customer:</b> I appreciate your dedication and support. I look forward to receiving the refund confirmation email soon. Your assistance is greatly appreciated. <br>
    <b>Agent:</b> It's our pleasure to assist you. We value your satisfaction, and we'll make sure the refund is processed efficiently. Should you require any further assistance or have any other questions, please don't hesitate to let us know. Have a wonderful day! <br>
    <b>Customer:</b> Thank you once again for your commitment. I hope the refund process goes smoothly. Have a great day too! <br>
    <b>Agent:</b> Thank you for your kind words. We're committed to resolving this issue to your satisfaction. If there's anything else we can assist you with, please don't hesitate to reach out. Take care and have a fantastic day ahead!
    </div>

    """
    st.markdown(transcript, unsafe_allow_html=True)

with st.expander("Example script for product support request"):
    transcript = """
    <div class="transcript">
    <b>Customer:</b> Hello, I recently purchased a laptop from your store, and I'm experiencing some technical difficulties. The laptop keeps randomly shutting down, and the battery life is significantly shorter than expected. I need assistance with troubleshooting and resolving these issues. <br>
    <b>Agent:</b> I apologize for the inconvenience you're facing with the laptop. I understand the frustration it must be causing. Thank you for bringing this to our attention. To assist you further, could you please provide me with your order number and a detailed description of the issues you're encountering? <br>
    <b>Customer:</b> Thank you for your prompt response. My order number is #987654321, and as mentioned, the laptop shuts down unexpectedly, even when the battery is charged, and the battery life lasts for only an hour, despite the advertised longer duration. <br>
    <b>Agent:</b> Thank you for sharing the details. I apologize for the inconvenience caused by the laptop's performance. To address these issues, I'll connect you with our technical support team, who will guide you through the troubleshooting process and help resolve the concerns. They will reach out to you shortly via email with further instructions and assistance. <br>
    <b>Customer:</b> I appreciate your assistance and prompt action. I look forward to hearing from the technical support team. I hope we can resolve these issues quickly. <br>
    <b>Agent:</b> We understand the importance of resolving these issues promptly for you. Our technical support team will work diligently to assist you and find a resolution. If you have any additional questions or need further support during the process, please don't hesitate to reach out to them. Thank you for your patience and cooperation. <br>
    <b>Customer:</b> Thank you for your dedication and support. I'll reach out to the technical support team as needed. I hope we can resolve these issues satisfactorily. <br>
    <b>Agent:</b> You're welcome. We're committed to ensuring your satisfaction and resolving these technical difficulties. Should you require any further assistance or have any other questions, please feel free to contact us. We appreciate your cooperation and look forward to resolving these issues for you.
    </div>

    """
    st.markdown(transcript, unsafe_allow_html=True)