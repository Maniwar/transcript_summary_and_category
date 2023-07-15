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
similarity_scores = []

for main_category, keywords in keywords_text.items():
    for keyword in keywords:
        keyword_embedding = keyword_embeddings[keyword.lower()]
        similarity_score = compute_semantic_similarity(keyword_embedding, comment_embedding)
        similarity_scores.append({'Keyword': keyword, 'Similarity Score': similarity_score})

# Create a DataFrame with similarity scores
df_similarity_scores = pd.DataFrame(similarity_scores)

# Get the top 10 items based on similarity score
top_10_items = df_similarity_scores.nlargest(10, 'Similarity Score')

# Display the top 10 items
st.subheader("Top 10 Items")
st.dataframe(top_10_items)


# Print similarity scores
st.subheader("Similarity Scores")
st.dataframe(similarity_scores)
