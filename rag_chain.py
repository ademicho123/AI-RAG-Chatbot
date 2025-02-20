from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from together import Together
from langchain.llms.base import LLM
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, PrivateAttr

load_dotenv()
class TogetherLLM(LLM, BaseModel):
    model_name: str = Field(default="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    api_key: str = Field(...)  # This makes api_key required
    _client: Together = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._client = Together(api_key=self.api_key)

    @property
    def _llm_type(self) -> str:
        return "together"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name
        }
class RAGChain:
    def __init__(self, vectorstore):
        # Load environment variables
        load_dotenv()
        self.vectorstore = vectorstore
        self.llm = self.get_llm()

    def get_llm(self):
        together_api_key = os.getenv("TOGETHER_API_KEY")
        print("Debug: Checking API key...")
        print(f"API Key present: {'Yes' if together_api_key else 'No'}")
        
        if not together_api_key:
            raise ValueError("TOGETHER_API_KEY is missing from .env file!")
        
        try:
            llm = TogetherLLM(api_key=together_api_key)
            print("Debug: LLM instance created successfully")
            return llm
        except Exception as e:
            print(f"Debug: Error creating LLM instance: {str(e)}")
        raise

    def create_chain(self):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
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