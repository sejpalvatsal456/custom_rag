import json
from urllib.request import Request, urlopen
from urllib.error import URLError

class Generator:
  def __init__(
    self, 
    model: str = "qwen3.5:4b", 
    base_url: str = "http://localhost:11434"
  ):
    self.model = model
    self.api_url = f"{base_url}/api/chat"
    
  def generate(self, query: str, context: list[dict]) -> str:
    
    if not query.strip():
      raise ValueError("Query can not be empty")
    if len(context) == 0:
      raise ValueError("Context can not be empty")
    
    context_text = ""
    
    for index, chunk in enumerate(context, start=1):
      context_text += (
        f"\n[Source: {index}]\n" + 
        f"Document: {chunk['source']}" + 
        f"Page: {chunk['page']}\n" + 
        f"Content: \n{chunk['text']}\n"
      )
    
    system_prompt = """
You are a document question-answering assistant.

Answer the user's question using ONLY the supplied document context.

Instructions:
1. Do not use outside knowledge to fill missing information.
2. If the context does not contain the answer, explicitly say so.
3. Do not invent facts, quotations, or page numbers.
4. Cite the relevant source and page number in your answer.
5. Be clear and concise.
"""
    user_prompt = f"""
DOCUMENT CONTEXT:
{context_text}

QUESTION:
{query}

Answer using the document context above.
"""
    payload = {
      "model": self.model,
      "messages": [
        {
          "role": "system",
          "content": system_prompt
        },
        {
          "role": "user",
          "content": user_prompt
        }
      ],
      "think": False,
      "stream": False,
      "think": False,
      "keep_alive": "10m",
      "options": {
        "num_ctx": 4096,
        "temperature": 0.2,
        "num_predict": 512
      }
    }
    
    req = Request(
      url=self.api_url,
      data= json.dumps(payload).encode("utf-8"),
      headers={
        "Content-Type": "application/json"
      },
      method="POST"
    )
    
    try:
      with urlopen(req, timeout=600) as res:
        result = json.loads(res.read().decode("utf-8"))
        # print("\n--- OLLAMA DEBUG ---")
        # print("Done reason:", result.get("done_reason"))
        # print("Message keys:", result.get("message", {}).keys())
        # print("Content:", repr(result["message"].get("content")))
        # print("Thinking:", repr(result["message"].get("thinking")))
        # print("Prompt tokens:", result.get("prompt_eval_count"))
        # print("Generated tokens:", result.get("eval_count"))
        # print("-------------------\n")

        answer = result["message"].get("content", "").strip()

        if not answer:
          raise RuntimeError(
            "Qwen returned an empty response. Check Ollama debug output."
          )
        return result["message"]["content"]
    except URLError as err:
      raise RuntimeError(f"Unexpected Error from Ollama: {err}") from err