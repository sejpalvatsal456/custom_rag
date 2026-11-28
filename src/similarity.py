import numpy as np

def cosine_similarity(query_vector: np.ndarray, document_vector: np.ndarray) -> np.ndarray:
  dot_product = np.dot(document_vector, query_vector)
  query_vector_norm = np.linalg.norm(query_vector)
  document_vector_norm = np.linalg.norm(document_vector, axis=1)
  norm_product = query_vector_norm*document_vector_norm
  similarity = np.divide(
    dot_product,
    norm_product,
    out = np.zeros_like(dot_product, dtype=float),
    where = norm_product != 0
  )
  
  return similarity

if __name__ == '__main__':
  query = np.array([1, 0, 0])

  documents = np.array([
      [1, 0, 0],
      [0, 1, 0],
      [-1, 0, 0]
  ])

  scores = cosine_similarity(query, documents)

  print(scores)