import logging
logging.basicConfig(level=logging.DEBUG, filename='debug.log', filemode='w')

from services import wiki
from services import llm

import config
config.LLM_PROVIDER = "ollama"

text = "In 2021, the Supreme Court ruled in Smith v. Jones (2021) 456 U.S. 123 that a late payment penalty of 5% is enforceable under Section 3 of the Contracts Act."

print("Running ingest...")
wiki._ingest_single_call(text, "test.pdf")
print("Done.")
