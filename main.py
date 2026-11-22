from pathlib import Path
from src.ingection.pdf_loader import load_pdf
from src.processing.chunker import chunk_documents

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
    
    
if __name__ == '__main__':
  main()