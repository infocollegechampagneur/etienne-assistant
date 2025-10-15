""" Étienne API - Assistant IA Éducatif pour le Collège Champagneur Version 2.0 - Optimisé pour Vercel (Sans ReportLab) """

from fastapi import FastAPI, HTTPException from fastapi.middleware.cors import CORSMiddleware from pydantic import BaseModel from typing import List, Optional from datetime import datetime import uuid

app = FastAPI( title="Étienne API - Assistant IA Éducatif", version="2.0", description="API pour l'assistant IA éducatif du Collège Champagneur" )

app.add_middleware( CORSMiddleware, allow_origins=[""], allow_credentials=True, allow_methods=[""], allow_headers=["*"], )

Modèles
class ChatRequest(BaseModel): message: str message_type: str session_id: Optional[str] = None

class TextAnalysisRequest(BaseModel): text: str analysis_type: str = "complete"

Sources
ENGLISH_SOURCES = { "grammar": [ "Oxford English Grammar (oxford.com)", "Cambridge Grammar (cambridge.org)", "BBC Learning English (bbc.co.uk/learningenglish)" ], "literature": [ "CliffsNotes Literature Guides (cliffsnotes.com)", "SparkNotes Literature (sparknotes.com)", "Lecturia Academic Library (lecturia.com)" ], "academic": [ "Purdue OWL Writing Lab (owl.purdue.edu)", "Harvard Writing Center (writingcenter.fas.harvard.edu)", "CliffsNotes Study Guides (cliffsnotes.com/study-guides)" ] }

QUEBEC_SOURCES = [ "Gouvernement du Québec (quebec.ca)", "MEES - Ministère de l'Éducation (education.gouv.qc.ca)", "Allô Prof (alloprof.qc.ca)" ]

Fonctions
def detect_language(text: str) -> str: english_words = ["the", "and", "to", "help", "english", "grammar"] french_words = ["le", "de", "et", "aide", "français"] text_lower = text.lower() words = text_lower.split() english_count = sum(1 for word in words if word in english_words) french_count = sum(1 for word in words if word in french_words) return "en" if english_count > french_count and english_count > 0 else "fr"

def detect_ai_content(text: str) -> dict: ai_indicators = ["as an ai", "i'm an ai", "as a language model"] text_lower = text.lower() ai_score = sum(0.2 for indicator in ai_indicators if indicator in text_lower) ai_probability = min(ai_score, 0.99) return { "ai_probability": round(ai_probability, 2), "is_likely_ai": ai_probability > 0.5, "confidence": "High" if ai_probability > 0.7 else "Medium" if ai_probability > 0.3 else "Low", "detected_patterns": [ind for ind in ai_indicators if ind in text_lower][:5] }

def check_plagiarism(text: str) -> dict: common_phrases = ["according to the study", "research shows"] text_lower = text.lower() score = sum(0.1 for phrase in common_phrases if phrase in text_lower) words = text_lower.split() diversity = len(set(words)) / len(words) if words else 0 if diversity < 0.6: score += 0.15 risk = min(score, 0.99) return { "plagiarism_risk": round(risk, 2), "is_suspicious": risk > 0.4, "vocabulary_diversity": round(diversity, 2), "risk_level": "High" if risk > 0.6 else "Medium" if risk > 0.3 else "Low", "found_phrases": [p for p in common_phrases if p in text_lower][:3], "suspicious_phrases": [p for p in common_phrases if p in text_lower][:3], "recommendation": "Vérifiez originalité" if risk > 0.4 else "Contenu semble original" }

def get_ai_response(message: str, message_type: str) -> dict: detected_lang = detect_language(message) message_lower = message.lower() is_english = any(w in message_lower for w in ["english", "grammar", "literature"])

if is_english:
    category = "literature" if "literature" in message_lower else "grammar"
    sources = ENGLISH_SOURCES[category][:3]
    if detected_lang == "en":
        response = f"For {category}, here are reliable sources:\n\n"
    else:
        response = f"Pour l'anglais ({category}):\n\n"
    for i, s in enumerate(sources, 1):
        response += f"{i}. {s}\n"
    trust = 0.95
else:
    sources = QUEBEC_SOURCES
    if detected_lang == "fr":
        response = "Bonjour ! Voici des ressources québécoises:\n\n"
    else:
        response = "Hello! Here are Quebec resources:\n\n"
    for i, s in enumerate(sources, 1):
        response += f"{i}. {s}\n"
    trust = 0.85

return {
    "response": response,
    "trust_score": trust,
    "sources": sources,
    "detected_language": detected_lang
}
Endpoints
@app.get("/") def root(): return {"message": "Étienne API"}

@app.get("/api") def api_root(): return { "message": "Étienne API - Assistant IA Éducatif", "status": "active", "platform": "vercel", "backend": "working", "version": "2.0", "college": "Collège Champagneur", "features": ["chat", "sources_anglaises", "detection_ia", "verification_plagiat"] }

@app.get("/api/health") def health(): return { "status": "healthy", "platform": "vercel", "version": "2.0" }

@app.post("/api/chat") def chat(request: ChatRequest): try: session_id = request.session_id or str(uuid.uuid4()) ai_result = get_ai_response(request.message, request.message_type)

    return {
        "id": str(uuid.uuid4()),
        "session_id": session_id,
        "message": request.message,
        "response": ai_result["response"],
        "message_type": request.message_type,
        "trust_score": ai_result["trust_score"],
        "sources": ai_result["sources"],
        "timestamp": datetime.utcnow().isoformat()
    }
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Erreur chat: {str(e)}")
@app.get("/api/subjects") def subjects(): return { "langues": { "name": "Langues", "subjects": ["Français", "Anglais", "Espagnol"] }, "sciences": { "name": "Sciences & Mathématiques", "subjects": ["Mathématiques", "Physique", "Chimie", "Biologie"] }, "sciences_humaines": { "name": "Sciences Humaines", "subjects": ["Histoire", "Géographie", "Économie"] }, "arts": { "name": "Arts", "subjects": ["Arts plastiques", "Musique", "Art dramatique"] } }

@app.post("/api/analyze-text") def analyze(request: TextAnalysisRequest): try: ai = detect_ai_content(request.text) plag = check_plagiarism(request.text) lang = detect_language(request.text)

    return {
        "text_length": len(request.text),
        "word_count": len(request.text.split()),
        "detected_language": lang,
        "ai_detection": ai,
        "plagiarism_check": plag,
        "overall_assessment": {
            "is_authentic": not ai["is_likely_ai"] and not plag["is_suspicious"],
            "recommendation": "Texte semble authentique" if not ai["is_likely_ai"] else "Vérification recommandée"
        }
    }
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Erreur analyse: {str(e)}")
@app.post("/api/detect-ai") def detect_ai(request: TextAnalysisRequest): try: return detect_ai_content(request.text) except Exception as e: raise HTTPException(status_code=500, detail=f"Erreur détection IA: {str(e)}")

@app.post("/api/check-plagiarism") def plagiarism(request: TextAnalysisRequest): try: return check_plagiarism(request.text) except Exception as e: raise HTTPException(status_code=500, detail=f"Erreur plagiat: {str(e)}")

handler = app
