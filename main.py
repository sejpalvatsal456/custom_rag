from pathlib import Path
from src.ingection.pdf_loader import load_pdf
from src.processing.chunker import chunk_documents
from src.embeddings.embedder import Embedder
from src.processing.retriever import Retriever
from src.vector_store.store import VectorStore
from src.generation.generator import Generator
from src.rag import RAG

PDF_DIR = Path("data/documents")

def show_result(result: dict):

  print("\nAnswer:")
  print(result["answer"])

  print("\nRetrieved sources:")

  for source in result["sources"]:
    print(
      f"- {source['source']} | "
      f"Page {source['page']} | "
      f"Similarity: {source['similarity']:.4f}"
          )

def main():
  store = VectorStore()
  chunks, embeddings = store.load()
  
  print(f"Loaded chunks: {len(chunks)}")
  print(f"Loaded embeddings: {embeddings.shape}")
  
  embedder = Embedder()
  retriever = Retriever(
    chunks=chunks,
    embeddings=embeddings,
    embedder=embedder
  )
  
  generator = Generator()
  
  rag = RAG(
    retriever=retriever,
    generator=generator
  )
  
  while True:
    query = input("\nAsk something (or type 'exit'): ")

    if query.lower().strip() == "exit":
      break
    
    try: 
      result = rag.ask(query=query)
      show_result(result)
      
    except Exception as e:
      print(f"Error: {e}")
    
if __name__ == '__main__':
  main()
  