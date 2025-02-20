from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
from together import Together
from langchain_core.runnables import Runnable
from typing import Dict, Any

load_dotenv()

class TogetherRunnable(Runnable):
    def __init__(self, api_key: str, model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"):
        self.client = Together(api_key=api_key)
        self.model = model

    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        query = input.get("query", "")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": query}],
        )
        return {"result": response.choices[0].message.content}

class RAGChain:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = self.get_llm()

    def get_llm(self):
        together_api_key = os.getenv("TOGETHER_API_KEY")
        if not together_api_key:
            raise ValueError("TOGETHER_API_KEY is missing from .env file!")
        
        # Initialize the custom Runnable
        return TogetherRunnable(api_key=together_api_key)

    def create_chain(self):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Create the RetrievalQA chain with the custom Runnable
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,  # Pass the custom Runnable here
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        return qa_chain

if __name__ == "__main__":
    from document_processor import DocumentProcessor
    from embedding_indexer import EmbeddingIndexer

    processor = DocumentProcessor("data/sample_text.txt")
    texts = processor.load_and_split()

    indexer = EmbeddingIndexer()
    vectorstore = indexer.create_vectorstore(texts)

    rag_chain = RAGChain(vectorstore)
    qa_chain = rag_chain.create_chain()

    query = "What is the capital of France?"
    result = qa_chain({"query": query})
    print(f"Answer: {result['result']}")