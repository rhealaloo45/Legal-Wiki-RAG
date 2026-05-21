import requests

prompt = """You are a legal wiki knowledge synthesizer. Read this document and create wiki pages that capture its legal meaning, statutory basis, precedents, and judicial reasoning.

PRINCIPLES:
- SOURCE INTEGRITY: DO NOT hallucinate or invent citations. Only cite cases, statutes,   or document names explicitly present in the text. This information comes from the   document 'test.pdf'. Explicitly mention the document name in your synthesis.
- FACTUAL PRECISION: DO NOT hallucinate dates or facts. If a date is not explicitly   stated, do not include it. Extract EXACT verbatim quotes for critical dates,   figures, and holdings.
- LEGAL DEPTH: Create pages for key precedents, statutory provisions, and the   judicial reasoning (ratio decidendi). Explain HOW the law was applied to the   facts, not just what the law is. Explain the Holding/Conclusion.
- Each page should read like a well-written wiki article.
- Include exact numbers, amounts, dates, rates, and timeframes verbatim.
- Flag contradictions or ambiguities you notice.

PAGE TITLES: Use specific, descriptive titles (e.g., "Ratio Decidendi: Late Payment Penalties", "Application of Section 3") — not generic ones like "Overview".

OUTPUT FORMAT — respond with valid JSON only, no explanation, no markdown fences:
{
  "pages": {
    "Descriptive Page Title": {
      "content": "4-10 sentence detailed synthesis with specific provisions, numbers, and conditions. Explain what it means and how it connects to other parts of the document.",
      "summary": "One-line summary of what this page covers."
    }
  },
  "relations": [
    {"from": "Page Title A", "to": "Page Title B", "label": "short verb phrase"}
  ]
}

Extract 10-30 pages and 10-40 relations. Cover the document thoroughly.

DOCUMENT:
In 2021, the Supreme Court ruled in Smith v. Jones (2021) 456 U.S. 123 that a late payment penalty of 5% is enforceable under Section 3 of the Contracts Act. The court reasoned that commercial certainty requires strict adherence to contract terms.
"""

url = "http://localhost:11434/api/generate"
payload = {
    "model": "llama3",
    "prompt": prompt,
    "stream": False,
    "options": {"temperature": 0.0} # Lower temp for determinism
}
resp = requests.post(url, json=payload, timeout=60)
print(repr(resp.json().get("response", "")))
