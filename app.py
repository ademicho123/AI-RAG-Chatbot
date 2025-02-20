import streamlit as st
from document_processor import DocumentProcessor
from embedding_indexer import EmbeddingIndexer
from rag_chain import RAGChain
from chatbot import Chatbot

@st.cache_resource
def initialize_chatbot(file_path):
    processor = DocumentProcessor(file_path)
    texts = processor.load_and_split()
    indexer = EmbeddingIndexer()
    vectorstore = indexer.create_vectorstore(texts)
    rag_chain = RAGChain(vectorstore)
    return Chatbot(rag_chain.create_chain())

st.title("AI RAG Chatbot")

# Update the file_uploader to accept PDF, DOC, and TXT files
uploaded_file = st.file_uploader(
    "Upload a text, PDF, or DOC file for the knowledge base", 
    type=["txt", "pdf", "doc", "docx"]
)

if uploaded_file:
    file_extension = uploaded_file.name.split('.')[-1]
    temp_file_path = f"temp_knowledge_base.{file_extension}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    chatbot = initialize_chatbot(temp_file_path)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    if prompt := st.chat_input("Ask a question"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = chatbot.get_response(prompt)
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.write("Please upload a text, PDF, or DOC file to start chatting!")