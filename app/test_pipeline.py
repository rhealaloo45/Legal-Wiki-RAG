import sys
import logging
logging.basicConfig(level=logging.DEBUG)

import config
from services import wiki
from services import llm

text = """In 2021, the Supreme Court ruled in Smith v. Jones (2021) 456 U.S. 123 that a late payment penalty of 5% is enforceable under Section 3 of the Contracts Act. The court reasoned that commercial certainty requires strict adherence to contract terms."""

print("Calling wiki._ingest_single_call...")
result = wiki._ingest_single_call(text, "test.pdf")
print("Result:")
print(result)
