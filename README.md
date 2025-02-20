# AI RAG Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot powered by LLaMA 3.3 70B Instruct Turbo using Together AI. It allows users to upload documents (`.txt`, `.pdf`, or `.docx`) to create a knowledge base and engage in context-aware conversations with the AI.

## Features
- Document Processing:
  - Supports multiple formats: `.txt`, `.pdf`, `.docx`
  - Automatic text chunking for optimal processing
  - Smart document parsing with metadata retention
- Vector Search:
  - FAISS-based vector storage for efficient similarity search
  - HuggingFace embeddings (all-MiniLM-L6-v2)
  - Fast and accurate context retrieval
- LLM Integration:
  - Powered by LLaMA 3.3 70B Instruct Turbo via Together AI
  - Context-aware responses using RAG
  - Maintains conversation history
- User Interface:
  - Clean, intuitive Streamlit interface
  - Real-time chat functionality
  - Easy document upload system

## Installation

1. Clone this repository:
```sh
git clone <repository-url>
cd <project-folder>
```

2. Create and activate a virtual environment (recommended):
```sh
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install the required dependencies:
```sh
pip install -r requirements.txt
```

## Configuration
1. Create a `.env` file in the project root directory
2. Add your Together AI API key:
```
TOGETHER_API_KEY=your_api_key_here
```

## Usage
1. Start the application:
```sh
streamlit run app.py
```

2. Open your web browser and navigate to the displayed URL (usually http://localhost:8501)

3. Upload a document using the file uploader

4. Start chatting with the AI about your document's content

## Project Structure
```
.
├── app.py                # Streamlit web interface
├── chatbot.py           # Core chatbot implementation
├── document_processor.py # Document handling and chunking
├── embedding_indexer.py  # Vector embedding and storage
├── rag_chain.py         # RAG pipeline implementation
├── requirements.txt     # Project dependencies
└── .env                # API key configuration
```

## Requirements
Python 3.8 or higher

## Dependencies
See requirements.txt for complete list of dependencies.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---