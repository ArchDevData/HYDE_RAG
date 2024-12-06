import openai
import streamlit as st
from io import StringIO

# Streamlit Secrets for Azure OpenAI
# GPT Model Configuration
GPT_API_KEY = st.secrets["GPT_API_KEY"]  # API key for GPT model
GPT_API_BASE = st.secrets["GPT_API_BASE"]  # Base URL for GPT model
GPT_API_VERSION = st.secrets["GPT_API_VERSION"]  # API version for GPT model
GPT_DEPLOYMENT_NAME = st.secrets["GPT_DEPLOYMENT_NAME"]  # Deployment name for GPT model

# Embedding Model Configuration
EMBEDDING_API_KEY = st.secrets["EMBEDDING_API_KEY"]  # API key for Embedding model
EMBEDDING_API_BASE = st.secrets["EMBEDDING_API_BASE"]  # Base URL for Embedding model
EMBEDDING_API_VERSION = st.secrets["EMBEDDING_API_VERSION"]  # API version for Embedding model
EMBEDDING_DEPLOYMENT_NAME = st.secrets["EMBEDDING_DEPLOYMENT_NAME"]  # Deployment name for Embedding model

# Streamlit UI
st.title("Hyde (Hypothetical Document Embeddings) App")
st.subheader("Upload multiple files for processing")

# File upload
uploaded_files = st.file_uploader(
    "Upload your files (TXT, PDF, CSV)", type=["txt", "pdf", "csv"], accept_multiple_files=True
)

def extract_text_from_file(uploaded_file):
    """Extract text from uploaded file (supports TXT and CSV)."""
    file_content = ""
    if uploaded_file.name.endswith(".txt"):
        file_content = uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".csv"):
        file_content = uploaded_file.read().decode("utf-8")
    else:
        st.warning(f"File type {uploaded_file.name} not supported.")
    return file_content

def process_with_hyde(file_content):
    """Generate hypothetical document and embeddings using Azure OpenAI."""
    try:
        # Configure GPT model
        openai.api_key = GPT_API_KEY
        openai.api_base = GPT_API_BASE
        openai.api_version = GPT_API_VERSION
        
        # Generate hypothetical document with GPT model
        hyde_prompt = f"Create a detailed hypothetical document based on the following content:\n\n{file_content}"
        response = openai.ChatCompletion.create(
            engine=GPT_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are an assistant for generating documents."},
                {"role": "user", "content": hyde_prompt},
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        synthetic_doc = response["choices"][0]["message"]["content"]
        
        # Configure Embedding model
        openai.api_key = EMBEDDING_API_KEY
        openai.api_base = EMBEDDING_API_BASE
        openai.api_version = EMBEDDING_API_VERSION
        
        # Generate embeddings for the document with the Embedding model
        embedding_response = openai.Embedding.create(
            engine=EMBEDDING_DEPLOYMENT_NAME,
            input=synthetic_doc,
        )
        embeddings = embedding_response["data"][0]["embedding"]

        return synthetic_doc, embeddings
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"Processing file: {uploaded_file.name}")
        file_content = extract_text_from_file(uploaded_file)
        
        if file_content:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                synthetic_doc, embeddings = process_with_hyde(file_content)
            
            if synthetic_doc:
                st.success(f"Processing complete for {uploaded_file.name}!")
                st.write("**Synthetic Document:**")
                st.text(synthetic_doc)
                st.write("**Embeddings (First 10 Values):**")
                st.write(embeddings[:10])
