from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import os

# Configuration pour Vercel
app = FastAPI(title="Étienne API", version="1.0.0")

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

# Sources anglaises
english_sources = {
    "literature": [
        "CliffsNotes Literature Guides (cliffsnotes.com)",
        "SparkNotes Literature (sparknotes.com)",
        "Project Gutenberg (gutenberg.org)"
    ],
    "grammar": [
        "Oxford English Grammar (oxford.com)",
        "Cambridge Grammar (cambridge.org)",
        "Purdue OWL (owl.purdue.edu)"
    ]
}

# Fonction IA simple
async def get_etienne_response(message: str, message_type: str) -> dict:
    try:
        message_lower = message.lower()
        
        # Détection anglais
        if any(word in message_lower for word in ["english", "shakespeare", "literature", "grammar"]):
            if "shakespeare" in message_lower or "literature" in message_lower:
                response = f"For English literature about '{message}', I recommend:\n1. CliffsNotes Literature Guides\n2. SparkNotes Literature\n3. Project Gutenberg\n\nThese provide comprehensive academic analysis."
                sources = english_sources["literature"]
            else:
                response = f"For English grammar about '{message}':\n1. Oxford English Grammar\n2. Cambridge Grammar\n3. Purdue OWL Writing Lab\n\nThese are the academic standards for English."
                sources = english_sources["grammar"]
            trust_score = 0.95
        else:
            # Réponse française
            response = f"Bonjour ! Je suis Étienne. Pour '{message}': Consultez les ressources québécoises officielles: 1) Sites .gouv.qc.ca 2) Universités québécoises 3) MEES education.gouv.qc.ca"
            sources = ["Sources éducatives québécoises"]
            trust_score = 0.85
        
        return {
            "response": response,
            "trust_score": trust_score,
            "sources": sources,
            "can_download": True
        }
    except Exception as e:
        return {
            "response": f"Erreur interne. Veuillez réessayer.",
            "trust_score": None,
            "sources": []
        }

# Routes
@app.get("/api")
async def root():
    return {
        "message": "Étienne API - Assistant IA Éducatif", 
        "status": "active",
        "platform": "vercel"
    }

@app.get("/api/health")
async def health():
    return {"status": "healthy", "assistant": "Étienne"}

@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        ai_result = await get_etienne_response(request.message, request.message_type)
        
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
        raise HTTPException(status_code=500, detail="Erreur de traitement")

@app.get("/api/subjects")
async def get_subjects():
    return {
        "langues": {"name": "Langues", "subjects": ["Français", "Anglais"]},
        "sciences": {"name": "Sciences", "subjects": ["Mathématiques", "Sciences"]},
        "arts": {"name": "Arts", "subjects": ["Arts plastiques", "Musique"]}
    }

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
