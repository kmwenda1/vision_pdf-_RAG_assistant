import os
import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import rag.retriever
from rag.vision import (
    validate_image,
    check_relevance_gate,
    retrieve_for_image,
    run_image_pipeline,
    IMAGE_RELEVANCE_THRESHOLD
)

def test_validate_image_types():
    valid_bytes = b"fake-data"
    
    # Valid types
    assert validate_image(valid_bytes, "test.png")[0] is True
    assert validate_image(valid_bytes, "test.jpg")[0] is True
    assert validate_image(valid_bytes, "test.webp")[0] is True
    
    # Invalid types
    is_valid, _, msg = validate_image(valid_bytes, "test.txt")
    assert is_valid is False
    assert "Unsupported file type" in msg

def test_validate_image_size():
    huge_bytes = b"0" * (11 * 1024 * 1024) # 11 MB
    is_valid, _, msg = validate_image(huge_bytes, "test.png")
    assert is_valid is False
    assert "maximum allowed is 10 MB" in msg

def test_check_relevance_gate_empty():
    passed, msg = check_relevance_gate([])
    assert passed is False
    assert "No documents are indexed yet" in msg

def test_check_relevance_gate_below_threshold():
    chunks = [{"score": IMAGE_RELEVANCE_THRESHOLD - 0.05, "text": "foo"}]
    passed, msg = check_relevance_gate(chunks)
    assert passed is False
    assert "I couldn't find this in your documents" in msg

def test_check_relevance_gate_above_threshold():
    chunks = [{"score": IMAGE_RELEVANCE_THRESHOLD + 0.10, "text": "bar"}]
    passed, msg = check_relevance_gate(chunks)
    assert passed is True
    assert msg == ""

@patch("rag.retriever.retrieve")
def test_retrieve_for_image_deduplication(mock_retrieve):
    # Mock return values for search queries
    mock_retrieve.side_effect = [
        [
            {"text": "Chunk A", "score": 0.4, "source": "doc1.pdf", "page_num": 1},
            {"text": "Chunk B", "score": 0.2, "source": "doc1.pdf", "page_num": 2},
        ],
        [
            {"text": "Chunk A", "score": 0.6, "source": "doc1.pdf", "page_num": 1}, # higher score for A
            {"text": "Chunk C", "score": 0.3, "source": "doc2.pdf", "page_num": 1},
        ],
    ]
    
    analysis = {"search_queries": ["query1", "query2"]}
    top_chunks = retrieve_for_image(analysis, "test question")
    
    assert len(top_chunks) == 3
    # Check that Chunk A kept the higher score (0.6)
    chunk_a = next(c for c in top_chunks if c["text"] == "Chunk A")
    assert chunk_a["score"] == 0.6

@patch("rag.vision.analyse_image")
@patch("rag.vision.retrieve_for_image")
@patch("rag.vision.generate_answer_with_image")
def test_run_image_pipeline_found(mock_gen, mock_ret, mock_analyse):
    mock_analyse.return_value = {
        "ocr_text": "Model X100",
        "description": "A photo of an equipment tag",
        "entities": ["X100"],
        "search_queries": ["Model X100 maintenance", "Model X100 specs"]
    }
    mock_ret.return_value = [
        {"text": "Maintenance schedule for Model X100", "score": 0.5, "source": "manual.pdf", "page_num": 4}
    ]
    mock_gen.return_value = {
        "answer": "Maintenance is required every 6 months according to page 4.",
        "sources": [{"source": "manual.pdf", "page_num": 4, "text": "Maintenance schedule...", "score": 0.5}],
        "model": "qwen/qwen3.8-27b",
        "error": None
    }
    
    result = run_image_pipeline(b"fake-bytes", "tag.jpg", "What is the maintenance schedule?")
    assert result["success"] is True
    assert result["confidence"] == "HIGH"
    assert "every 6 months" in result["answer"]
    assert len(result["sources"]) == 1

@patch("rag.vision.analyse_image")
@patch("rag.vision.retrieve_for_image")
def test_run_image_pipeline_not_found_gate(mock_ret, mock_analyse):
    mock_analyse.return_value = {
        "ocr_text": "Alien Tech 999",
        "description": "An unknown device",
        "entities": ["Alien Tech"],
        "search_queries": ["Alien Tech 999"]
    }
    # Chunks below threshold
    mock_ret.return_value = [
        {"text": "Unrelated topic", "score": 0.02, "source": "manual.pdf", "page_num": 1}
    ]
    
    result = run_image_pipeline(b"fake-bytes", "alien.jpg", "Is this covered?")
    assert result["success"] is True
    assert result["confidence"] == "GATE_BLOCKED"
    assert "I couldn't find this in your documents" in result["answer"]
    assert "An unknown device" in result["answer"]
    assert result["sources"] == []


@patch("time.sleep", return_value=None)
@patch("rag.vision._get_gemini_client")
def test_analyse_image_retry_success(mock_get_client, mock_sleep):
    """Test vision client fails with 503 twice, then succeeds on 3rd attempt."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    import google.genai.errors as errors
    err_503 = errors.APIError(503, "High demand - unavailable", None)

    mock_resp = MagicMock()
    mock_resp.text = '{"ocr_text": "QR-440X", "description": "Pressure washer tag", "entities": ["QR-440X"], "search_queries": ["QR-440X manual"]}'

    mock_client.models.generate_content.side_effect = [err_503, err_503, mock_resp]

    from rag.vision import analyse_image
    result = analyse_image(b"fake-bytes", "image/jpeg", "What model is this?")

    assert "error" not in result
    assert result["ocr_text"] == "QR-440X"
    assert mock_client.models.generate_content.call_count == 3
    assert mock_sleep.call_count == 2


@patch("time.sleep", return_value=None)
@patch("rag.vision._get_gemini_client")
def test_analyse_image_all_retries_fail(mock_get_client, mock_sleep):
    """Test vision client returns 503 on every attempt and returns friendly error without crashing."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    import google.genai.errors as errors
    err_503 = errors.APIError(503, "High demand - unavailable", None)
    mock_client.models.generate_content.side_effect = err_503

    from rag.vision import analyse_image, run_image_pipeline
    analysis = analyse_image(b"fake-bytes", "image/jpeg", "What model is this?")

    assert "error" in analysis
    assert "user_friendly_error" in analysis
    assert "busy right now" in analysis["user_friendly_error"]

    # Also verify run_image_pipeline returns friendly error message to UI
    pipeline_res = run_image_pipeline(b"fake-bytes", "test.jpg", "What model is this?")
    assert pipeline_res["success"] is False
    assert "The image analysis service is busy right now. Please try again in a minute." in pipeline_res["answer"]
    assert "503" in pipeline_res["error"]


@patch("time.sleep", return_value=None)
@patch("rag.vision._get_gemini_client")
def test_analyse_image_non_transient_fails_immediately(mock_get_client, mock_sleep):
    """Test that HTTP 400/401/403 fails immediately without retrying."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    import google.genai.errors as errors
    err_400 = errors.ClientError(400, "Bad Request - Invalid image format", None)
    mock_client.models.generate_content.side_effect = err_400

    from rag.vision import analyse_image
    analysis = analyse_image(b"fake-bytes", "image/jpeg", "What model is this?")

    assert "error" in analysis
    # Must fail on the first attempt without calling sleep
    assert mock_client.models.generate_content.call_count == 1
    assert mock_sleep.call_count == 0

