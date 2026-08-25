"""OCR transcription tests for POST /api/upload-file (handwritten student copy).

QUOTA NOTE: Gemini free tier -> this module makes at most 1 OCR call
(session-scoped fixture) against /tmp/manuscrit_eleve.jpg.

Covers, per the 2nd fix (PIL preprocessing + rebalanced 2-step prompt):
  C1 completeness, C2 fidelity, C3 no teacher annotations, C4 contract/latency.
"""
import os
import re
import json
import time

import pytest
import requests
from dotenv import dotenv_values

frontend_env = dotenv_values("/app/frontend/.env")
base_url = os.environ.get("REACT_APP_BACKEND_URL") or frontend_env.get("REACT_APP_BACKEND_URL")
if not base_url:
    raise RuntimeError("REACT_APP_BACKEND_URL missing")
BASE_URL = base_url.rstrip("/")

IMAGE_PATH = "/tmp/manuscrit_eleve.jpg"
DUMP_PATH = "/app/test_reports/ocr_manuscrit_response.json"


@pytest.fixture(scope="session")
def ocr_response():
    """Single OCR upload call (quota limited)."""
    if not os.path.exists(IMAGE_PATH):
        pytest.skip("Test image /tmp/manuscrit_eleve.jpg missing")
    started = time.time()
    with open(IMAGE_PATH, "rb") as fh:
        resp = requests.post(
            f"{BASE_URL}/api/upload-file",
            files={"file": ("manuscrit_eleve.jpg", fh, "image/jpeg")},
            timeout=180,
        )
    elapsed = time.time() - started
    try:
        payload = resp.json()
    except Exception:
        payload = {"_raw": resp.text[:2000]}
    os.makedirs(os.path.dirname(DUMP_PATH), exist_ok=True)
    with open(DUMP_PATH, "w", encoding="utf-8") as out:
        json.dump({"status_code": resp.status_code, "elapsed_s": round(elapsed, 2),
                   "body": payload}, out, ensure_ascii=False, indent=2)
    print(f"\n--- OCR call: status={resp.status_code} elapsed={elapsed:.1f}s ---")
    if isinstance(payload, dict) and payload.get("extracted_text"):
        print(payload["extracted_text"])
    return resp.status_code, payload, elapsed


# --- C4: response contract + latency (preprocessing must not break anything) ---
def test_status_schema_and_latency(ocr_response):
    status, body, elapsed = ocr_response
    assert status == 200, f"Expected 200, got {status}: {body}"
    assert isinstance(body, dict)
    assert "extracted_text" in body, f"Missing extracted_text: {list(body)}"
    assert isinstance(body["extracted_text"], str)
    assert body.get("filename") == "manuscrit_eleve.jpg"
    assert "_id" not in body
    assert elapsed < 60, f"OCR took {elapsed:.1f}s (>60s budget)"


# --- C1: completeness ---
def test_completeness_word_and_line_count(ocr_response):
    _, body, _ = ocr_response
    text = body["extracted_text"]
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    lines = [ln for ln in text.splitlines() if ln.strip()]
    print(f"words={len(words)} lines={len(lines)}")
    assert len(words) >= 90, f"Transcription incomplete: only {len(words)} words"
    assert len(lines) >= 12, f"Transcription incomplete: only {len(lines)} non-empty lines"


def test_completeness_start_and_end_present(ocr_response):
    _, body, _ = ocr_response
    low = body["extracted_text"].lower()
    assert "en 2001" in low, "Beginning of the text ('En 2001') missing"
    assert re.search(r"gens\s+qu'?il\s+tue", low), "End of the text ('gens qu'il tue') missing"


# --- C2: fidelity, student spelling mistakes must be preserved ---
EXPECTED_MISSPELLINGS = ["neu", "rilhouette", "abendonen", "obscuriter",
                         "appeller", "creature", "causer"]
FORBIDDEN_CORRECTIONS = ["new york", "silhouette", "abandonné", "abandonner",
                         "obscurité", "appelé", "créature", "causée"]


def test_misspellings_preserved(ocr_response):
    _, body, _ = ocr_response
    text_low = body["extracted_text"].lower()
    found = [w for w in EXPECTED_MISSPELLINGS
             if re.search(r"\b" + re.escape(w), text_low)]
    missing = [w for w in EXPECTED_MISSPELLINGS if w not in found]
    print(f"Preserved misspellings {len(found)}/7: {found} | missing: {missing}")
    assert len(found) >= 6, f"Only {len(found)}/7 misspellings preserved. Missing: {missing}"


def test_no_silent_corrections(ocr_response):
    _, body, _ = ocr_response
    text_low = body["extracted_text"].lower()
    corrected = [w for w in FORBIDDEN_CORRECTIONS if w in text_low]
    print(f"Corrections detected: {corrected}")
    assert not corrected, f"AI silently corrected student spelling: {corrected}"


# --- C3: teacher red-ink annotation codes must be ignored ---
ANNOTATION_CODES = [r"\bG1\b", r"\bG4\b", r"\bG5\b", r"\bU1\b", r"\bU2\b", r"Titre\s*\?"]


def test_teacher_annotations_excluded(ocr_response):
    _, body, _ = ocr_response
    text = body["extracted_text"]
    present = [c for c in ANNOTATION_CODES if re.search(c, text)]
    print(f"Annotation codes leaked: {present}")
    assert not present, f"Teacher annotation codes present in transcription: {present}"
