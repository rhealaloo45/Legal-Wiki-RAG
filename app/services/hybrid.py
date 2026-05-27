import logging
from services import llm

logger = logging.getLogger(__name__)


def generate_answer(question: str, rag_context: str, chunks: list, wiki_context: str, titles: list) -> dict:
    """Generate an answer using both RAG and Wiki context."""
    if not chunks and not wiki_context:
        return {
            "answer": "Both the RAG excerpts and Wiki are empty. No context available to answer.",
            "usage": {}
        }

    # Combine both contexts into a single block, then use the shared prompt
    combined_parts = []
    if wiki_context:
        combined_parts.append("=== SYNTHESIZED WIKI PAGES ===\n" + wiki_context)
    if rag_context:
        combined_parts.append("=== RAW EXCERPTS (RAG) ===\n" + rag_context)

    combined_context = "\n\n".join(combined_parts) if combined_parts else "No context available."

    from services.prompts import ANSWER_PROMPT
    prompt = ANSWER_PROMPT.format(context=combined_context, question=question)

    usage = {}
    try:
        answer, usage = llm.ask(prompt, pipeline="hybrid")
        import re
        answer = re.sub(r'<reasoning>.*?</reasoning>', '', answer, flags=re.DOTALL).strip()
    except RuntimeError as e:
        answer = f"⚠️ LLM error: {e}"

    return {
        "answer": answer,
        "chunks_used": len(chunks),
        "pages_used": len(titles),
        "usage": usage
    }
