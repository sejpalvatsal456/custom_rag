import json
from pathlib import Path
import numpy as np

class VectorStore:
  def __init__(self, storage_path: str = "data/embeddings"):
    self.storage_path = Path(storage_path)
    self.storage_path.mkdir(
      parents=True,
      exist_ok=True
    )
    self.vector_path = self.storage_path / "vectors.npy"
    self.chunks_path = self.storage_path / "chunk.json"
    
  def save(self, chunks: list[dict], embeddings: np.ndarray):
    
    if len(chunks) != len(embeddings):
      raise ValueError("The Chunks and Embeddings can't have different dimensions")
    
    np.save(self.vector_path, embeddings)
    
    with open(self.chunks_path, "w", encoding="utf-8") as file:
      json.dump(chunks, file, ensure_ascii=False, indent=2)
      
    print("Data saved successfully")
    
  def load(self) -> tuple[list[dict], np.ndarray]:
    
    if not self.vector_path.exists():
      raise FileNotFoundError("The file of vector storage does not exist.")
    
    if not self.chunks_path.exists():
      raise FileNotFoundError("The file of chunks storage does not exist.")
    
    embeddings = np.load(self.vector_path)
    
    with open(self.chunks_path, "r", encoding="utf-8") as file:
      chunks = json.load(file)
      
    if len(chunks) != len(embeddings):
      raise ValueError("The Chunks and Embeddings can't have different dimensions")
      
    return chunks, embeddings
    