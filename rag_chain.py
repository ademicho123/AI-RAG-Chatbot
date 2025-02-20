from langchain.chains import RetrievalQA
from langchain.llms import OpenAI, Gemini, Fireworks
from dotenv import load_dotenv
import os

load_dotenv()

class RAGChain:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = self.get_llm()

    def get_llm(self):
        if os.getenv("OPENAI_API_KEY"):
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
        elif os.getenv("GEMINI_API_KEY"):
            return Gemini(api_key=os.getenv("GEMINI_API_KEY"), temperature=0)
        elif os.getenv("FIREWORKS_API_KEY"):
            return Fireworks(api_key=os.getenv("FIREWORKS_API_KEY"), temperature=0)
        else:
            raise ValueError("No valid API key found! Please set one in .env file.")

    def create_chain(self):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
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