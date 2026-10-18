# ─────────────────────────────────────────────────────────
# rag/generator.py — Answer Generation via Groq
#
# Copied from insurance-rag and kept identical.
# Sends the retrieved chunks + question to Groq (Llama/Qwen)
# and returns a cited answer.
# ─────────────────────────────────────────────────────────

from groq import Groq
from rag.config import get_secret

_groq_client = None

def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        key = get_secret("GROQ_API_KEY")
        if not key:
            raise EnvironmentError("GROQ_API_KEY is not set. Add it to Streamlit Secrets or your .env file.")
        _groq_client = Groq(api_key=key)
    return _groq_client

MODEL = "qwen/qwen3.8-27b"

SYSTEM_PROMPT = """You are a precise document assistant.
Your job is to answer questions based ONLY on the provided document excerpts.

STRICT RULES:
1. Only use information from the provided context. Never use outside knowledge.
2. For every statement you make, cite the source: mention the document name and page number.
3. If the answer is not in the context, say exactly: "I could not find this information in the provided documents."
4. Do not guess or fill in gaps with plausible-sounding information.
5. If numbers (amounts, limits, dates) appear in the context, quote them exactly as written.
6. Keep your answer clear and professional. Use bullet points for lists.
7. At the end of your answer, list the sources you used.

Remember: Be precise or say you don't know."""


def generate_answer(query: str, retrieved_chunks: list[dict]) -> dict:
    """
    Generates a cited answer from retrieved chunks.

    Parameters:
        query:            the user's question
        retrieved_chunks: list of chunk dicts from retriever.py

    Returns:
        {"answer": "...", "sources": [...], "model": "...", "error": None}
    """
    if not retrieved_chunks:
        return {
            "answer":  "I could not find any relevant information in the uploaded documents.",
            "sources": [],
            "model":   MODEL,
            "error":   None,
        }

    context_parts = [
        f"[Source: {c['source']}, Page {c['page_num']}]\n{c['text']}"
        for c in retrieved_chunks
    ]
    context_block = "\n\n---\n\n".join(context_parts)

    user_message = f"""Here are the relevant excerpts from the documents:

{context_block}

Based ONLY on the above excerpts, please answer this question:

{query}"""

    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0,
            max_tokens=1024,
        )
        return {
            "answer":  response.choices[0].message.content,
            "sources": retrieved_chunks,
            "model":   MODEL,
            "error":   None,
        }

    except Exception as e:
        err = str(e)
        if "api_key" in err.lower() or "authentication" in err.lower():
            friendly = "API key error: check GROQ_API_KEY in your .env file."
        elif "rate_limit" in err.lower():
            friendly = "Rate limit reached. Wait a moment and try again."
        else:
            friendly = f"API error: {err}"

        return {
            "answer":  f"Sorry, I couldn't generate an answer. {friendly}",
            "sources": retrieved_chunks,
            "model":   MODEL,
            "error":   friendly,
        }


def format_sources(chunks: list[dict]) -> str:
    """Formats source chunks into a readable string."""
    if not chunks:
        return "No sources available."
    seen, lines = set(), []
    for c in chunks:
        key = f"{c['source']} — Page {c['page_num']}"
        if key not in seen:
            lines.append(f"• {key}")
            seen.add(key)
    return "\n".join(lines)
