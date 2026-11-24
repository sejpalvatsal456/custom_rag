from pathlib import Path
from src.ingection.pdf_loader import load_pdf
from src.processing.chunker import chunk_documents
from src.embeddings.embedder import Embedder
from src.processing.retriever import Retriever

PDF_DIR = Path("data/documents")

def show_result(results: list[dict]):
  for result in results:

    print("\n------------------------------")

    print("Source:", result["source"])
    print("Page:", result["page"])
    print("Similarity:", round(result["similarity"], 4))

    print("Content:")
    print(result["text"])

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
  
  retriever = Retriever(chunks, embeddings, embedder)
  result = retriever.retrieve(
    query="How does GPT use gossip and rumors in storytelling?",
    top_k=5
  )
  
  show_result(result)
    
    
if __name__ == '__main__':
  main()