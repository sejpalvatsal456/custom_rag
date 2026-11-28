def split_text(
  text: str,
  chunk_size: int = 1000,
  overlap: int = 150
) -> list[str] :
  if chunk_size <= 0:
    raise ValueError("chunk_size must be positive.")

  if overlap < 0 or overlap >= chunk_size:
    raise ValueError("overlap must be between 0 and chunk_size.")

  chunks = []
  start = 0
  
  while start < len(text):
    end = min(start + chunk_size, len(text))
    
    if end < len(text):
      boundary = text.rfind(" ", start, end)
      if boundary > start:
        end = boundary
        
    chunk = text[start:end].strip()
    
    if chunk:
      chunks.append(chunk)
    
    if end >= len(text): 
      break
    
    start = end - overlap
    
  return chunks

def chunk_documents(
  documents: list[dict],
  chunk_size: int = 1000,
  overlap: int = 150
) -> list[dict]:
  all_chunks = []
  
  for doc in documents:
    text_chunks = split_text(doc["text"], chunk_size, overlap)
    
    for chunk_index, chunk in enumerate(text_chunks): 
      all_chunks.append({
        "source": doc['source'],
        "page": doc["page"],
        "chunk_id": chunk_index,
        "text": chunk
      })
  
  return all_chunks