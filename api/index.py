from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ENGLISH_SOURCES = {
    "grammar": ["Oxford English Grammar (oxford.com)", "BBC Learning English (bbc.co.uk)"],
    "literature": ["CliffsNotes (cliffsnotes.com)", "SparkNotes (sparknotes.com)"]
}

QUEBEC_SOURCES = ["Gouvernement du Québec (quebec.ca)", "Allô Prof (alloprof.qc.ca)"]

def detect_language(text: str) -> str:
    english_words = ["the", "and", "help", "english", "grammar"]
    french_words = ["le", "de", "aide", "français"]
    text_lower = text.lower()
    eng = sum(1 for w in english_words if w in text_lower)
    fr = sum(1 for w in french_words if w in text_lower)
    return "en" if eng > fr else "fr"

def detect_ai(text: str) -> dict:
    patterns = ["as an ai", "language model"]
    text_lower = text.lower()
    score = sum(0.3 for p in patterns if p in text_lower)
    prob = min(score, 0.99)
    return {
        "ai_probability": round(prob, 2),
        "is_likely_ai": prob > 0.5,
        "confidence": "High" if prob > 0.7 else "Low"
    }

def check_plag(text: str) -> dict:
    phrases = ["according to", "research shows"]
    text_lower = text.lower()
    score = sum(0.2 for p in phrases if p in text_lower)
    risk = min(score, 0.99)
    return {
        "plagiarism_risk": round(risk, 2),
        "is_suspicious": risk > 0.4,
        "risk_level": "High" if risk > 0.6 else "Low",
        "suspicious_phrases": [p for p in phrases if p in text_lower]
    }

@app.get("/")
@app.get("/api")
async def root():
    return {
        "message": "Étienne API - Assistant IA Éducatif",
        "status": "active",
        "platform": "vercel",
        "backend": "working",
        "version": "2.0",
        "features": ["chat", "sources_anglaises", "detection_ia", "verification_plagiat"]
    }

@app.get("/api/health")
async def health():
    return {"status": "healthy", "version": "2.0"}

@app.post("/api/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        message = body.get("message", "")
        lang = detect_language(message)
        is_eng = "english" in message.lower() or "grammar" in message.lower()
        
        if is_eng:
            sources = ENGLISH_SOURCES["grammar"]
            response = "For English, here are reliable sources:\n\n"
            for i, s in enumerate(sources, 1):
                response += f"{i}. {s}\n"
            trust = 0.95
        else:
            sources = QUEBEC_SOURCES
            response = "Bonjour ! Voici des ressources québécoises:\n\n"
            for i, s in enumerate(sources, 1):
                response += f"{i}. {s}\n"
            trust = 0.85
        
        return {
            "id": str(uuid.uuid4()),
            "session_id": body.get("session_id", str(uuid.uuid4())),
            "message": message,
            "response": response,
            "message_type": body.get("message_type", "je_veux"),
            "trust_score": trust,
            "sources": sources,
            "timestamp": "2025-01-15T12:00:00"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/subjects")
async def subjects():
    return {
        "langues": {"name": "Langues", "subjects": ["Français", "Anglais"]},
        "sciences": {"name": "Sciences", "subjects": ["Mathématiques", "Physique"]}
    }

@app.post("/api/analyze-text")
async def analyze(request: Request):
    try:
        body = await request.json()
        text = body.get("text", "")
        ai = detect_ai(text)
        plag = check_plag(text)
        lang = detect_language(text)
        return {
            "text_length": len(text),
            "word_count": len(text.split()),
            "detected_language": lang,
            "ai_detection": ai,
            "plagiarism_check": plag
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect-ai")
async def detect_ai_endpoint(request: Request):
    try:
        body = await request.json()
        return detect_ai(body.get("text", ""))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/check-plagiarism")
async def plagiarism(request: Request):
    try:
        body = await request.json()
        return check_plag(body.get("text", ""))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

handler = app
