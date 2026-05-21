import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "llama3",
    "prompt": "You are a helpful assistant. Output a JSON object with keys 'foo' and 'bar'. Do not output any other text.",
    "stream": False,
}
resp = requests.post(url, json=payload, timeout=10)
print(repr(resp.json().get("response", "")))
