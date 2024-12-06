# Hyde (Hypothetical Document Embeddings) App

This Streamlit app allows you to:
- Upload multiple files (TXT, PDF, or CSV).
- Process the files with **Azure OpenAI** to generate:
  - Hypothetical documents.
  - Embeddings for downstream tasks.

## Features
1. **File Uploads**: Supports TXT, CSV, and basic text-based file formats.
2. **Azure OpenAI Integration**:
   - Uses `gpt-35-turbo` for document generation.
   - Uses `text-embedding-ada-002` for embedding generation.
3. **Dynamic Results**: Displays processed synthetic documents and embeddings in the app.

## How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/hyde-app.git
   cd hyde-app
