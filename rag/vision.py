# ─────────────────────────────────────────────────────────
# rag/vision.py — Image → Answer Pipeline  (NEW)
#
# WHAT THIS FILE DOES:
#   Bridges an uploaded image into the text-based RAG.
#   The image is NEVER embedded or searched directly.
#   We convert it to text first, then reuse retriever + generator.
#
# PIPELINE:
#   image + question
#     → validate_image()      → check type/size, return mime-type
#     → analyse_image()       → Gemini vision → structured JSON
#     → retrieve_for_image()  → multi-query retrieval, deduplicate
#     → relevance gate        → bail if nothing clears threshold
#     → generate_answer_with_image() → existing generator + image context
#
# NEW DEPENDENCY: google-genai (vision only).
#   Groq (existing LLM) has no vision models on this account.
#   Gemini 3.8-flash supports image input and is confirmed working.
# ─────────────────────────────────────────────────────────

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types as genai_types
from rag.config import get_secret

# ── Tuneable constants ────────────────────────────────────────────────────────

# Vision model configuration (configurable via Streamlit Secrets or environment variables)
VISION_MODEL: str = get_secret("VISION_MODEL", "gemini-3.8-flash")
VISION_MODEL_FALLBACK: str = get_secret("VISION_MODEL_FALLBACK", "gemini-3.5-flash")

# Relevance gate threshold.
# retriever score = 1 - cosine_distance  (higher = more similar).
# Image-derived queries score slightly lower than verbatim matches,
# so we gate at 0.10 (below the existing LOW threshold of 0.15).
# Override with IMAGE_RELEVANCE_THRESHOLD env var or Streamlit secret.
IMAGE_RELEVANCE_THRESHOLD: float = float(
    get_secret("IMAGE_RELEVANCE_THRESHOLD", "0.10")
)

# Max chunks kept after merging results from all search queries
IMAGE_TOP_K: int = int(get_secret("IMAGE_TOP_K", "6"))

# Number of search queries the vision model generates per image
IMAGE_QUERY_COUNT: int = 3

# 10 MB — safe upper limit for Gemini API
MAX_IMAGE_BYTES: int = 10 * 1024 * 1024

ALLOWED_MIME_TYPES: dict[str, str] = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
}

# ── Debug logging (set DEBUG_VISION=1 in env to enable) ──────────────────────

logger = logging.getLogger(__name__)


def _debug(msg: str) -> None:
    if os.getenv("DEBUG_VISION", "0") == "1":
        logger.debug(msg)
        print(f"[vision DEBUG] {msg}")


# ── Gemini client (lazy singleton) ────────────────────────────────────────────

_gemini_client: Optional[genai.Client] = None


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        api_key = get_secret("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Please configure it in Streamlit Secrets or your .env file."
            )
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


# ── Step 0: Validate ──────────────────────────────────────────────────────────

def validate_image(image_bytes: bytes, filename: str) -> tuple[bool, str, str]:
    """
    Checks file type and size.

    Returns (is_valid, mime_type, error_message).
    is_valid=True  → mime_type is populated, error_message is "".
    is_valid=False → mime_type is "", error_message explains the problem.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_MIME_TYPES:
        return False, "", (
            f"Unsupported file type '{suffix}'. "
            f"Allowed types: jpg, png, webp."
        )
    if len(image_bytes) > MAX_IMAGE_BYTES:
        size_mb = len(image_bytes) / (1024 * 1024)
        return False, "", (
            f"Image is {size_mb:.1f} MB — maximum allowed is 10 MB."
        )
    return True, ALLOWED_MIME_TYPES[suffix], ""


# ── Step 1: Analyse image with Gemini vision ──────────────────────────────────

_ANALYSIS_PROMPT = """\
You are an expert document analyst. Analyse the attached image in the context \
of the user's question and return a single JSON object with exactly these keys:

{{
  "ocr_text": "<all visible text, numbers, codes, labels verbatim — empty string if none>",
  "description": "<1-2 sentence factual description of what the image shows>",
  "entities": ["<key identifier 1>", "<key identifier 2>", ...],
  "search_queries": ["<query 1>", "<query 2>", "<query 3>"]
}}

Rules:
- ocr_text: copy every word, number, and code exactly as printed.
- description: factual only, no guessing.
- entities: model numbers, error codes, policy references, product names.
- search_queries: write exactly {query_count} short queries (5-10 words each) \
that would find the answer to the user's question inside an indexed PDF document. \
Base them on BOTH the image content AND the question.

User's question: {question}

Respond with ONLY the JSON. No markdown fences, no extra text."""


def _is_transient_error(exc: Exception) -> tuple[bool, Optional[int]]:
    """
    Checks if an exception represents a transient error that should be retried.
    Transient: HTTP 503, 429, 500, or network timeouts / connection errors.
    Non-transient: HTTP 400, 401, 403, 404, or invalid image errors.
    """
    import re
    # Check google.genai error code or status_code
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        if code in (503, 429, 500):
            return True, code
        if code in (400, 401, 403, 404):
            return False, code

    err_str = str(exc).lower()

    # Match HTTP status codes in error message string if not set as attribute
    match = re.search(r"\b(503|429|500|400|401|403|404)\b", err_str)
    if match:
        status = int(match.group(1))
        if status in (503, 429, 500):
            return True, status
        if status in (400, 401, 403, 404):
            return False, status

    # Network timeouts, unavailable, rate limits, connection errors
    if any(k in err_str for k in ("unavailable", "high demand", "rate limit", "timeout", "timed out", "connection error", "connection reset", "econnreset")):
        return True, 503 if "unavailable" in err_str or "high demand" in err_str else 429

    return False, None


def analyse_image(image_bytes: bytes, mime_type: str, question: str) -> dict:
    """
    Sends image + question to Gemini and returns structured analysis JSON.

    On success: dict with ocr_text, description, entities, search_queries.
    On failure: dict with 'error' and 'user_friendly_error' keys.

    Resilient:
    - Retries transient errors (503, 429, 500, network timeouts) up to 4 attempts.
    - Exponential backoff ~1s, 2s, 4s + random jitter.
    - Fails immediately on 400/401/403.
    - Configurable fallback model (VISION_MODEL_FALLBACK).
    - Logs attempt details when Debug Mode is enabled.
    """
    import time
    import random

    client = _get_gemini_client()
    prompt = _ANALYSIS_PROMPT.format(
        query_count=IMAGE_QUERY_COUNT,
        question=question,
    )

    # Determine models to try: primary first, then fallback if configured
    models_to_try = [VISION_MODEL]
    if VISION_MODEL_FALLBACK and VISION_MODEL_FALLBACK != VISION_MODEL:
        models_to_try.append(VISION_MODEL_FALLBACK)

    max_attempts = 4
    base_delays = [1.0, 2.0, 4.0]  # wait before attempt 2, 3, 4
    last_err: Optional[Exception] = None
    last_code: Optional[int] = None

    for model in models_to_try:
        _debug(f"Starting image analysis with model: {model}")
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[
                        genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        prompt,
                    ],
                )
                raw = response.text.strip()
                _debug(f"Vision raw response (model {model}, attempt {attempt}):\n{raw}")

                # Strip accidental markdown fences
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]

                analysis = json.loads(raw)
                analysis.setdefault("ocr_text", "")
                analysis.setdefault("description", "")
                analysis.setdefault("entities", [])
                analysis.setdefault("search_queries", [])

                _debug(f"Parsed analysis successfully on attempt {attempt}: {json.dumps(analysis, indent=2)}")
                return analysis

            except json.JSONDecodeError as exc:
                return {
                    "error": f"Vision model returned invalid JSON: {exc}. Raw: {raw[:200]}",
                    "user_friendly_error": "Failed to parse image details. Please try another photo."
                }
            except Exception as exc:
                last_err = exc
                is_transient, code = _is_transient_error(exc)
                last_code = code

                if not is_transient:
                    # Fail immediately on non-transient errors (400, 401, 403, etc.)
                    _debug(f"Non-transient error on attempt {attempt} (model {model}, error_code={code}): {exc}")
                    return {
                        "error": f"Vision model call failed: {exc}",
                        "user_friendly_error": f"Image analysis failed ({code or 'invalid request'}). Please verify your image and settings."
                    }

                # Transient error: compute wait time if more attempts remain
                if attempt < max_attempts:
                    base_delay = base_delays[attempt - 1]
                    jitter = random.uniform(0.1, 0.5)
                    wait_time = base_delay + jitter
                    _debug(
                        f"Attempt {attempt}/{max_attempts} failed for model '{model}' with error code {code}. "
                        f"Waiting {wait_time:.2f}s before attempt {attempt + 1}."
                    )
                    time.sleep(wait_time)
                else:
                    _debug(f"All {max_attempts} attempts failed for model '{model}' with error code {code}.")

    # Everything failed across all configured models
    return {
        "error": f"Vision model call failed: {last_err}",
        "user_friendly_error": "The image analysis service is busy right now. Please try again in a minute."
    }


# ── Step 2: Multi-query retrieval + deduplication ─────────────────────────────

def retrieve_for_image(analysis: dict, user_question: str, collection_name: Optional[str] = None) -> list[dict]:
    """
    Runs each search_query through the existing retrieve() function,
    merges results, deduplicates by chunk text (keeping the best score),
    and returns the top IMAGE_TOP_K chunks from collection_name.

    Falls back to the raw user question if no queries were generated.
    """
    from rag.retriever import retrieve  # existing function — not modified

    queries = analysis.get("search_queries", []) or [user_question]
    _debug(f"Running {len(queries)} retrieval queries: {queries}")

    seen: dict[str, dict] = {}  # chunk text → best chunk dict

    for query in queries:
        results = retrieve(query, top_k=IMAGE_TOP_K, collection_name=collection_name)
        _debug(f"  '{query}' → {len(results)} results")
        for chunk in results:
            key = chunk["text"]
            if key not in seen or chunk["score"] > seen[key]["score"]:
                seen[key] = chunk
                _debug(
                    f"    chunk score={chunk['score']:.3f} "
                    f"src={chunk['source']} pg={chunk['page_num']}"
                )

    merged = sorted(seen.values(), key=lambda c: c["score"], reverse=True)
    top = merged[:IMAGE_TOP_K]
    _debug(f"After merge+dedup: {len(top)} chunks")
    return top


# ── Step 3: Relevance gate ────────────────────────────────────────────────────

def check_relevance_gate(chunks: list[dict]) -> tuple[bool, str]:
    """
    Prevents hallucination: if no chunk clears the threshold, we
    do NOT call the answer model. Returns (passed, message).

    passed=True  → proceed to answer generation
    passed=False → return the message to the user instead of an answer
    """
    if not chunks:
        return False, "No documents are indexed yet. Please upload a PDF first."

    best = chunks[0]["score"]
    _debug(f"Relevance gate: best={best:.3f}, threshold={IMAGE_RELEVANCE_THRESHOLD}")

    if best < IMAGE_RELEVANCE_THRESHOLD:
        return False, (
            f"I couldn't find this in your documents "
            f"(best match score: {best:.2f}, minimum needed: {IMAGE_RELEVANCE_THRESHOLD:.2f})."
        )
    return True, ""


# ── Step 4: Answer generation with image context ──────────────────────────────

def generate_answer_with_image(
    user_question: str,
    chunks: list[dict],
    analysis: dict,
) -> dict:
    """
    Calls the existing generate_answer() with an enriched question that
    includes the image's OCR text, description, and entities.

    generator.py is NOT modified — we just pass it a richer query string.
    """
    from rag.generator import generate_answer  # existing function — not modified

    ocr_text    = analysis.get("ocr_text", "").strip()
    description = analysis.get("description", "").strip()
    entities    = analysis.get("entities", [])

    image_ctx_parts = []
    if description:
        image_ctx_parts.append(f"Image shows: {description}")
    if ocr_text:
        image_ctx_parts.append(f"Text visible in image: {ocr_text}")
    if entities:
        image_ctx_parts.append(f"Key identifiers: {', '.join(str(e) for e in entities)}")

    if image_ctx_parts:
        enriched_question = (
            f"{user_question}\n\n"
            f"[Context extracted from the uploaded image]\n"
            + "\n".join(image_ctx_parts)
            + "\n\n"
            "Answer ONLY from the provided document chunks. "
            "Cite the source document and page number for every claim. "
            "If the chunks do not contain the answer, say so explicitly."
        )
    else:
        enriched_question = user_question

    _debug(f"Enriched question:\n{enriched_question}")
    return generate_answer(enriched_question, chunks)


# ── Public entry point ────────────────────────────────────────────────────────

def run_image_pipeline(
    image_bytes: bytes,
    filename: str,
    user_question: str,
    collection_name: Optional[str] = None,
) -> dict:
    """
    Full image → answer pipeline. Called from app.py.

    Parameters:
        image_bytes:     raw bytes of the uploaded image
        filename:        original filename (for mime-type detection)
        user_question:   the user's text question (required alongside the image)
        collection_name: optional per-session collection name

    Returns:
        {
          "success":    bool,
          "answer":     str,   # generated answer or gate/error message
          "sources":    list,  # chunk dicts (empty on failure/gate)
          "confidence": str,   # "HIGH" / "MEDIUM" / "LOW" / "NONE" / "GATE_BLOCKED"
          "warning":    str,
          "analysis":   dict,  # ocr_text, description, entities, search_queries
          "debug_info": dict,  # intermediate outputs (visible when DEBUG_VISION=1)
          "error":      str,   # non-empty on hard failures
        }
    """
    debug_info: dict = {}

    # ── 0. Validate ──────────────────────────────────────────────────────────
    valid, mime_type, err = validate_image(image_bytes, filename)
    if not valid:
        return {
            "success": False, "answer": f"❌ {err}", "sources": [],
            "confidence": "NONE", "warning": "", "analysis": {},
            "debug_info": debug_info, "error": err,
        }

    # ── 1. Analyse image ─────────────────────────────────────────────────────
    analysis = analyse_image(image_bytes, mime_type, user_question)
    debug_info["image_analysis"] = analysis

    if "error" in analysis:
        debug_info["error"] = analysis["error"]
        friendly = analysis.get("user_friendly_error", "The image analysis service is busy right now. Please try again in a minute.")
        return {
            "success": False,
            "answer": f"❌ {friendly}",
            "sources": [], "confidence": "NONE", "warning": "",
            "analysis": analysis, "debug_info": debug_info,
            "error": analysis["error"],
        }

    # ── 2. Multi-query retrieval ──────────────────────────────────────────────
    chunks = retrieve_for_image(analysis, user_question, collection_name=collection_name)
    debug_info["retrieved_chunks"] = [
        {"source": c["source"], "page_num": c["page_num"], "score": c["score"]}
        for c in chunks
    ]

    # ── 3. Relevance gate ─────────────────────────────────────────────────────
    passed, gate_msg = check_relevance_gate(chunks)
    if not passed:
        gate_answer = gate_msg
        if analysis.get("description"):
            gate_answer += f"\n\n**What the image showed:** {analysis['description']}"
        if analysis.get("ocr_text"):
            gate_answer += f"\n\n**Text visible in image:** {analysis['ocr_text']}"
        return {
            "success": True,   # pipeline ran — just no match
            "answer": gate_answer,
            "sources": [], "confidence": "GATE_BLOCKED",
            "warning": "⚠️ No relevant information found in the indexed documents for this image.",
            "analysis": analysis, "debug_info": debug_info, "error": "",
        }

    # ── 4. Generate answer ────────────────────────────────────────────────────
    generation = generate_answer_with_image(user_question, chunks, analysis)
    debug_info["generation"] = {
        "model": generation.get("model"),
        "error": generation.get("error"),
    }

    # Confidence label (same thresholds as retriever.retrieve_with_confidence)
    top_score = chunks[0]["score"] if chunks else 0
    if top_score > 0.25:
        confidence, warning = "HIGH", None
    elif top_score > 0.15:
        confidence, warning = "MEDIUM", "⚠️ Moderate confidence — verify against the source document."
    else:
        confidence, warning = "LOW", "⚠️ Low confidence. The image topic may not be fully covered in the indexed documents."

    return {
        "success":    True,
        "answer":     generation["answer"],
        "sources":    generation["sources"],
        "confidence": confidence,
        "warning":    warning or generation.get("error", ""),
        "analysis":   analysis,
        "debug_info": debug_info,
        "error":      generation.get("error", ""),
    }
