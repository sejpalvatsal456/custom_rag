from pathlib import Path
import pymupdf

def load_pdf(file_path: str) -> list[dict]:
  path = Path(file_path)
  
  if not path.exists():
    raise FileNotFoundError(f"The file {path} not exists.")
  
  if path.suffix.lower() != ".pdf":
    raise ValueError(f"The file {path} is not pdf.")
  
  documents = []
  
  with pymupdf.open(path) as pdf:
    for page_number, page in enumerate(pdf, start=1):
      # print(page_number)
      text = page.get_text("text").strip()
      if not text: 
        continue
      # print(text)
      
      documents.append({
          "source": path.name,
          "page": page_number,
          "text": text
      })
  
  return documents
      
  
if __name__ == '__main__':
  documents = load_pdf("D:/Code Files/Python/custom_rag/data/documents/StoryScope.pdf")
  print(documents)