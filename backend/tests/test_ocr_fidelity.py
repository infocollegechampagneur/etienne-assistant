"""OCR diplomatic transcription fidelity test for POST /api/upload-file.

QUOTA NOTE: Gemini free tier -> this module makes at most 1 OCR call
(session-scoped fixture) against /tmp/manuscrit_eleve.jpg.
"""
import os
import re
import json

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
    with open(IMAGE_PATH, "rb") as fh:
        resp = requests.post(
            f"{BASE_URL}/api/upload-file",
            files={"file": ("manuscrit_eleve.jpg", fh, "image/jpeg")},
            timeout=180,
        )
    payload = None
    try:
        payload = resp.json()
    except Exception:
        payload = {"_raw": resp.text[:2000]}
    os.makedirs(os.path.dirname(DUMP_PATH), exist_ok=True)
    with open(DUMP_PATH, "w", encoding="utf-8") as out:
        json.dump({"status_code": resp.status_code, "body": payload}, out,
                  ensure_ascii=False, indent=2)
    return resp.status_code, payload


# --- Response contract ---
def test_status_and_schema(ocr_response):
    status, body = ocr_response
    assert status == 200, f"Expected 200, got {status}: {body}"
    assert isinstance(body, dict)
    assert "extracted_text" in body, f"Missing extracted_text: {list(body)}"
    assert isinstance(body["extracted_text"], str)
    assert body.get("filename") == "manuscrit_eleve.jpg"
    assert "_id" not in body


def test_text_non_empty_and_multiline(ocr_response):
    _, body = ocr_response
    text = body["extracted_text"]
    assert len(text.strip()) > 100, f"Text too short ({len(text)} chars)"
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) >= 6, f"Expected ~10-13 lines of text, got {len(lines)}"


# --- Fidelity: student spelling mistakes must be preserved ---
EXPECTED_MISSPELLINGS = ["neu", "rilhouette", "abendonen", "obscuriter",
                         "appeller", "creature", "causer"]
FORBIDDEN_CORRECTIONS = ["new york", "silhouette", "abandonné", "abandonner",
                         "obscurité", "appelé", "créature", "causée"]


def test_misspellings_preserved(ocr_response):
    _, body = ocr_response
    text_low = body["extracted_text"].lower()
    found = [w for w in EXPECTED_MISSPELLINGS
             if re.search(r"\b" + re.escape(w), text_low)]
    missing = [w for w in EXPECTED_MISSPELLINGS if w not in found]
    print(f"Preserved misspellings {len(found)}/7: {found} | missing: {missing}")
    assert len(found) >= 5, f"Only {len(found)}/7 misspellings preserved. Missing: {missing}"


def test_no_silent_corrections(ocr_response):
    _, body = ocr_response
    text_low = body["extracted_text"].lower()
    corrected = [w for w in FORBIDDEN_CORRECTIONS if w in text_low]
    print(f"Corrections detected: {corrected}")
    assert not corrected, f"AI silently corrected student spelling: {corrected}"


# --- Teacher red-ink annotation codes must be ignored ---
ANNOTATION_CODES = [r"\bG1\b", r"\bG4\b", r"\bG5\b", r"\bU1\b", r"\bU2\b", r"Titre\s*\?"]


def test_teacher_annotations_excluded(ocr_response):
    _, body = ocr_response
    text = body["extracted_text"]
    present = [c for c in ANNOTATION_CODES if re.search(c, text)]
    print(f"Annotation codes leaked: {present}")
    assert not present, f"Teacher annotation codes present in transcription: {present}"
