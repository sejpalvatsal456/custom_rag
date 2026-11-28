from pathlib import Path
from src.pdf_loader import load_pdf
from src.chunker import chunk_documents
from src.embedder import Embedder
from src.retriever import Retriever
from src.vector_store.store import VectorStore
from src.generator import Generator
from src.rag import RAG
from src.vector_store.indexer import DocumentIndexer

PDF_DIR = Path("data/documents")

def show_sync_summary(summary: dict):
  print(
    f"Sync done | added: {len(summary['added'])}, "
    f"modified: {len(summary['modified'])}, "
    f"deleted: {len(summary['deleted'])} | "
    f"{summary['total_documents']} documents, {summary['total_chunks']} chunks"
  )

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
  # store = VectorStore()
  # chunks, embeddings = store.load()
  
  indexer = DocumentIndexer()
  summary = indexer.sync()
  show_sync_summary(summary)
  chunks, embeddings = indexer.store.load()
  
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
  
  # store = VectorStore()
  # existing_chunks, existing_embeddings = store.load()
  # print(f"Existing Chunks: {len(existing_chunks)}")
  # print(f"Existing Embeddings: {len(existing_embeddings)}")
  
  # existing_chunk_count = {}
  # for chunk in existing_chunks:
  #   source = chunk['source']
  #   existing_chunk_count[source] = existing_chunk_count.get(source, 0) + 1
    
  # print(existing_chunk_count)