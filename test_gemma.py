import requests
import json

prompt = """
You are a legal wiki knowledge synthesizer. Read this document excerpt and produce a JSON response.
OUTPUT FORMAT — respond with valid JSON only, no explanation, no markdown fences:
{
  "pages": {
    "Document Overview": {
      "content": "Detailed 6-12 sentence summary.",
      "summary": "One-line summary."
    }
  },
  "relations": []
}

DOCUMENT EXCERPT:
This is a test legal document about a contract dispute.
"""

payload = {
    "model": "gemma",
    "prompt": prompt,
    "stream": False
}
resp = requests.post("http://localhost:11434/api/generate", json=payload)
data = resp.json()
print("RAW OUTPUT:")
print(data.get("response", ""))
