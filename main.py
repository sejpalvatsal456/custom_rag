from pathlib import Path
from src.ingection.pdf_loader import load_pdf
from src.processing.chunker import chunk_documents
from src.embeddings.embedder import Embedder

PDF_DIR = Path("data/documents")

def main():
  pdf_files = list(PDF_DIR.glob("*.pdf"))
  # print(pdf_files)
  
  if not pdf_files:
    print("No PDF files found.")
    return

  all_docs = []

  for file in pdf_files:
    print(f"Loading file: {file.name}")
    documents = load_pdf(str(file))
    # print(documents)
    
    all_docs.extend(documents)
  
  # print(len(all_docs))
  chunks = chunk_documents(
    all_docs,
    chunk_size=1000,
    overlap=150
  )
  
  print(f"Chunks: {len(chunks)}")
  
  embedder = Embedder()
  
  texts = [chunk['text'] for chunk in chunks]
  
  embeddings = embedder.generate_embedding(texts)
  print("\nEmbedding generation completed!")

  print("Number of embeddings:", embeddings.shape[0])
  print("Dimensions per embedding:", embeddings.shape[1])

  print("\nFirst embedding:")
  print(embeddings[0].shape)
    
    
if __name__ == '__main__':
  main()