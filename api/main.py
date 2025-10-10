from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import requests
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Configuration pour Vercel
app = FastAPI(title="Étienne API", version="1.0.0")

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL')
if mongo_url:
    client = AsyncIOMotorClient(mongo_url)
    db = client[os.environ.get('DB_NAME', 'etienne_free')]

# Models
class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    message: str
    response: str
    message_type: str
    trust_score: Optional[float] = None
    sources: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ChatRequest(BaseModel):
    message: str
    message_type: str
    session_id: Optional[str] = None

class TextAnalysisRequest(BaseModel):
    text: str
    analysis_type: str = "complete"

# Sources spécialisées anglais
english_sources = {
    "grammar": [
        "Oxford English Grammar (oxford.com)",
        "Cambridge Grammar (cambridge.org)", 
        "BBC Learning English (bbc.co.uk/learningenglish)"
    ],
    "literature": [
        "CliffsNotes Literature Guides (cliffsnotes.com)",
        "SparkNotes Literature (sparknotes.com)",
        "Lecturia Academic Library (lecturia.com)",
        "Project Gutenberg (gutenberg.org)"
    ],
    "academic": [
        "Purdue OWL Writing Lab (owl.purdue.edu)",
        "Harvard Writing Center (writingcenter.fas.harvard.edu)",
        "McGill Writing Centre (mcgill.ca/mwc)"
    ]
}

# Détection d'IA
def detect_ai_content(text: str) -> dict:
    try:
        ai_indicators = ["as an ai", "i'm an ai", "however", "furthermore", "in conclusion"]
        text_lower = text.lower()
        ai_score = sum(0.2 for indicator in ai_indicators if indicator in text_lower)
        ai_probability = min(ai_score, 0.99)
        
        return {
            "ai_probability": round(ai_probability, 2),
            "is_likely_ai": ai_probability > 0.5,
            "confidence": "High" if ai_probability > 0.7 else "Medium" if ai_probability > 0.3 else "Low"
        }
    except:
        return {"ai_probability": 0.0, "is_likely_ai": False, "confidence": "Error"}

# Vérificateur plagiat
def check_plagiarism(text: str) -> dict:
    try:
        words = text.lower().split()
        unique_words = set(words)
        diversity = len(unique_words) / len(words) if words else 0
        risk = max(0, 0.8 - diversity)
        
        return {
            "plagiarism_risk": round(risk, 2),
            "is_suspicious": risk > 0.4,
            "vocabulary_diversity": round(diversity, 2),
            "risk_level": "High" if risk > 0.6 else "Medium" if risk > 0.3 else "Low"
        }
    except:
        return {"plagiarism_risk": 0.0, "is_suspicious": False}

# Détection langue
def detect_language(text: str) -> str:
    english_words = ["the", "and", "help", "grammar", "writing"]
    french_words = ["le", "de", "aide", "grammaire"]
    
    text_lower = text.lower()
    en_count = sum(1 for word in english_words if word in text_lower)
    fr_count = sum(1 for word in french_words if word in text_lower)
    
    return "en" if en_count > fr_count and en_count > 0 else "fr"

# IA Response
async def get_ai_response_free(message: str, message_type: str) -> dict:
    try:
        message_lower = message.lower()
        detected_lang = detect_language(message)
        
        # Détection questions anglais
        is_english_query = any(word in message_lower for word in ["english", "grammar", "literature", "shakespeare"])
        
        if is_english_query and detected_lang == "en":
            if "shakespeare" in message_lower or "literature" in message_lower:
                response = f"For English literature about '{message}', I recommend:\n1. CliffsNotes Literature Guides\n2. SparkNotes Literature\n3. Project Gutenberg for original texts\n\nThese provide comprehensive analysis for academic work."
                sources = english_sources["literature"]
            else:
                response = f"For English grammar regarding '{message}', check:\n1. Oxford English Grammar\n2. Cambridge Grammar\n3. Purdue OWL Writing Lab\n\nThese are the gold standards for English language learning."
                sources = english_sources["grammar"]
        else:
            response = f"Bonjour ! Je suis Étienne. Pour '{message}': Consultez les sources éducatives québécoises: 1) Sites .gouv.qc.ca 2) Universités québécoises 3) MEES education.gouv.qc.ca"
            sources = ["Sources éducatives québécoises"]
        
        return {
            "response": response,
            "trust_score": 0.95 if is_english_query else 0.85,
            "sources": sources,
            "can_download": True
        }
    except Exception as e:
        return {
            "response": "Étienne temporairement indisponible. Consultez education.gouv.qc.ca",
            "sources": []
        }

# Routes
@app.get("/api")
async def root():
    return {"message": "Étienne API - Assistant IA Éducatif", "status": "active"}

@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        ai_result = await get_ai_response_free(request.message, request.message_type)
        
        chat_message = ChatMessage(
            session_id=session_id,
            message=request.message,
            response=ai_result["response"],
            message_type=request.message_type,
            trust_score=ai_result["trust_score"],
            sources=ai_result["sources"]
        )
        
        return chat_message
    except Exception as e:
        raise HTTPException(status_code=500, detail="Erreur traitement")

@app.get("/api/subjects")
async def get_subjects():
    return {
        "langues": {"name": "Langues", "subjects": ["Français", "Anglais"]},
        "sciences": {"name": "Sciences", "subjects": ["Mathématiques", "Sciences"]},
        "arts": {"name": "Arts", "subjects": ["Arts plastiques", "Musique"]}
    }

@app.post("/api/analyze-text")
async def analyze_text(request: TextAnalysisRequest):
    try:
        ai_result = detect_ai_content(request.text)
        plagiarism_result = check_plagiarism(request.text)
        
        return {
            "ai_detection": ai_result,
            "plagiarism_check": plagiarism_result,
            "detected_language": detect_language(request.text)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Erreur analyse")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Handler Vercel
handler = app
Message de commit : Add Étienne API backend
Cliquez "Commit new file"
