# AI RAG Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot powered by LLaMA 3.3 70B using Together AI. It allows users to upload `.txt`, `.pdf`, or `.docx` files to create a knowledge base and chat with the AI.

## Features
- Supports multiple document formats: `.txt`, `.pdf`, `.docx`
- Uses FAISS for efficient vector search
- Powered by LLaMA 3.3 70B via Together AI
- Real-time interactive chat with document-based retrieval

## Installation
Clone this repository and install the required dependencies:
```sh
git clone <repository-url>
cd <project-folder>
pip install -r requirements.txt

## Usage
1. **Set up API Keys**  
   Create a `.env` file and add your Together AI API key:
   ```sh
   TOGETHER_API_KEY=your_api_key_here
   ```
   
2. **Run the Chatbot**  
   ```sh
   streamlit run app.py
   ```
   Upload a document and start chatting!

## File Structure
```
- app.py                # Streamlit frontend
- chatbot.py            # Chatbot logic
- document_processor.py # Handles document loading and processing
- embedding_indexer.py  # Vector indexing with FAISS
- rag_chain.py          # RAG pipeline setup
- requirements.txt      # Dependencies
```

## Dependencies
- `langchain`
- `streamlit`
- `faiss-cpu`
- `together`
- `python-dotenv`
- `sentence_transformers`
- `PyMuPDF` (for PDFs)
- `python-docx` (for Word documents)

## License
This project is licensed under the MIT License.
```

