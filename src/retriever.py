import numpy as np
from src.embedder import Embedder
from src.similarity import cosine_similarity

class Retriever:
  def __init__(self, chunks: list[dict], embeddings: np.ndarray, embedder: Embedder):
    
    if len(chunks) != len(embeddings):
      raise ValueError("The dimensions of chunks and embedding are not matching.")
    
    self.chunks = chunks
    self.embeddings = embeddings
    self.embedder = embedder
    
  def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
    if not query.strip():
      raise ValueError("Query can not be empty.")
    
    if top_k <= 0:
      raise ValueError("top_k should be positive.")
    
    query_embedding = self.embedder.generate_embedding([query])[0]
    similarity = cosine_similarity(
      query_vector=query_embedding,
      document_vector=self.embeddings
    )
    top_k_indexes = np.argsort(similarity)[::-1][:top_k]
    results = []
    for index in top_k_indexes:
      results.append({
        **self.chunks[index],
        "similarity": float(similarity[index])
      })
    return results