import streamlit as st
import pandas as pd
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import textwrap

# Function to chunk large text
def chunk_text(text, max_length):
    return textwrap.wrap(text, max_length)

# Function to summarize text
def summarize_text(text, model, tokenizer, max_length=512):
    prompt = "Summarize the key points: " + text
    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, length_penalty=2.0, num_return_sequences=1, temperature=0.7)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Streamlit Interface
st.title('Comments Summarization')
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if 'Comments' in data.columns:
        all_comments = " ".join(data['Comments'].fillna(""))

        # Chunk the document
        chunks = chunk_text(all_comments, 800)  # Adjust chunk size as needed

        # Initialize the GPT-Neo model and tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

        # Summarize each chunk and combine
        with st.spinner("Generating summary..."):
            summaries = [summarize_text(chunk, model, tokenizer) for chunk in chunks]

            # Iterative summarization for a more concise final summary
            combined_summary = " ".join(summaries)
            final_summary = summarize_text(combined_summary, model, tokenizer)

            st.subheader("Executive Summary")
            st.write(final_summary)
    else:
        st.error("The CSV file does not have a 'Comments' column.")
else:
    st.write("Please upload a CSV file.")
