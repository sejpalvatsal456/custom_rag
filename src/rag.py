from src.processing.retriever import Retriever
from src.generation.generator import Generator

class RAG:
  def __init__(self, retriever: Retriever, generator: Generator):
    self.retriever = retriever
    self.generator = generator
    
  def ask(self, query: str, top_k: int = 5) -> dict:
    
    retrieved_chunks = self.retriever.retrieve(query, top_k)
    answer = self.generator.generate(query, context=retrieved_chunks)
    return {
      "query": query,
      "answer": answer,
      "sources": [
        {
          "source": chunk['source'],
          "page": chunk['page'],
          "similarity": chunk['similarity']
        }
        for chunk in retrieved_chunks
      ]
    }