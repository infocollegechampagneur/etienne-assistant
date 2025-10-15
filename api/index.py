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
import json
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

class DocumentRequest(BaseModel):
    content: str
    title: str = "Document Étienne"
    format: str = "pdf"

class TextAnalysisRequest(BaseModel):
    text: str
    analysis_type: str = "complete"

# Sources crédibles spécialisées pour l'anglais (mises à jour)
english_sources = {
    "grammar": [
        "Oxford English Grammar (oxford.com)",
        "Cambridge Grammar (cambridge.org)", 
        "Grammarly Blog (grammarly.com/blog)",
        "BBC Learning English (bbc.co.uk/learningenglish)",
        "British Council (learnenglish.britishcouncil.org)",
        "Merriam-Webster Dictionary (merriam-webster.com)"
    ],
    "literature": [
        "Project Gutenberg (gutenberg.org)",
        "Poetry Foundation (poetryfoundation.org)",
        "CliffsNotes Literature Guides (cliffsnotes.com)",
        "SparkNotes Literature (sparknotes.com)",
        "Lecturia Academic Library (lecturia.com)",
        "Norton Anthology Online (wwnorton.com)",
        "Oxford Literature Online (oxfordliteratureonline.com)"
    ],
    "academic": [
        "Purdue OWL Writing Lab (owl.purdue.edu)",
        "Harvard Writing Center (writingcenter.fas.harvard.edu)",
        "MIT Writing Center (cmsw.mit.edu/writing-and-communication-center)",
        "University of Toronto Writing Centre (writing.utoronto.ca)",
        "McGill Writing Centre (mcgill.ca/mwc)",
        "UBC Writing Centre (students.ubc.ca/academic-success/writing-centre)",
        "CliffsNotes Study Guides (cliffsnotes.com/study-guides)"
    ],
    "esl": [
        "BBC Learning English (bbc.co.uk/learningenglish)",
        "British Council (learnenglish.britishcouncil.org)",
        "English Central (englishcentral.com)",
        "Perfect English Grammar (perfect-english-grammar.com)",
        "FluentU English (fluentu.com/blog/english)",
        "EnglishClub (englishclub.com)"
    ]
}

# Détecteur d'IA et vérificateur de plagiat
def detect_ai_content(text: str) -> dict:
    """Détecte si un texte a été généré par IA"""
    try:
        # Indicateurs d'IA (patterns communs)
        ai_indicators = [
            "as an ai", "i'm an ai", "as a language model", "i don't have personal",
            "i cannot", "i can't provide", "it's important to note", "however",
            "furthermore", "moreover", "in conclusion", "to summarize"
        ]
        
        text_lower = text.lower()
        ai_score = 0
        detected_patterns = []
        
        # Vérification des patterns d'IA
        for indicator in ai_indicators:
            if indicator in text_lower:
                ai_score += 0.2
                detected_patterns.append(indicator)
        
        # Vérification de la structure (phrases très uniformes)
        sentences = text.split('.')
        if len(sentences) > 3:
            avg_length = sum(len(s.strip().split()) for s in sentences if s.strip()) / len([s for s in sentences if s.strip()])
            if avg_length > 15:  # Phrases longues et complexes
                ai_score += 0.1
        
        # Score final
        ai_probability = min(ai_score, 0.99)
        
        return {
            "ai_probability": round(ai_probability, 2),
            "is_likely_ai": ai_probability > 0.5,
            "confidence": "High" if ai_probability > 0.7 else "Medium" if ai_probability > 0.3 else "Low",
            "detected_patterns": detected_patterns[:3]  # Top 3 patterns
        }
        
    except Exception as e:
        return {
            "ai_probability": 0.0,
            "is_likely_ai": False,
            "confidence": "Error",
            "error": str(e)
        }

def check_plagiarism(text: str) -> dict:
    """Vérificateur de plagiat basique"""
    try:
        # Phrases communes qui peuvent indiquer du plagiat
        common_academic_phrases = [
            "according to the study", "research shows that", "studies have shown",
            "it has been proven that", "experts agree that", "the data suggests",
            "furthermore", "in addition", "however", "therefore", "consequently"
        ]
        
        # Vérification de phrases trop parfaites/académiques
        text_lower = text.lower()
        academic_score = 0
        found_phrases = []
        
        for phrase in common_academic_phrases:
            if phrase in text_lower:
                academic_score += 0.1
                found_phrases.append(phrase)
        
        # Vérification de la diversité du vocabulaire
        words = text_lower.split()
        unique_words = set(words)
        vocabulary_diversity = len(unique_words) / len(words) if words else 0
        
        # Score de risque de plagiat
        plagiarism_risk = min(academic_score, 0.9)
        if vocabulary_diversity < 0.6:  # Faible diversité = risque
            plagiarism_risk += 0.1
        
        plagiarism_risk = min(plagiarism_risk, 0.99)
        
        return {
            "plagiarism_risk": round(plagiarism_risk, 2),
            "is_suspicious": plagiarism_risk > 0.4,
            "vocabulary_diversity": round(vocabulary_diversity, 2),
            "risk_level": "High" if plagiarism_risk > 0.6 else "Medium" if plagiarism_risk > 0.3 else "Low",
            "found_phrases": found_phrases[:3],
            "recommendation": "Vérifiez l'originalité avec des sources académiques" if plagiarism_risk > 0.4 else "Contenu semble original"
        }
        
    except Exception as e:
        return {
            "plagiarism_risk": 0.0,
            "is_suspicious": False,
            "error": str(e)
        }

# Détection de langue pour réponses adaptées
def detect_language(text: str) -> str:
    """Détecte la langue du message"""
    english_words = [
        "the", "and", "to", "of", "a", "in", "is", "it", "you", "that", "he", "was", "for", "on", "are", "as", "with",
        "help", "me", "can", "could", "would", "should", "what", "how", "where", "when", "why", "grammar", "writing"
    ]
    
    french_words = [
        "le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir", "que", "pour", "dans", "ce", "son", "une", "sur",
        "aide", "moi", "peux", "pourrais", "voudrais", "devrais", "quoi", "comment", "où", "quand", "pourquoi", "grammaire"
    ]
    
    text_lower = text.lower()
    words = text_lower.split()
    
    english_count = sum(1 for word in words if word in english_words)
    french_count = sum(1 for word in words if word in french_words)
    
    # Si plus de mots anglais détectés
    if english_count > french_count and english_count > 0:
        return "en"
    else:
        return "fr"

# IA Gratuite Hugging Face pour Étienne
async def get_ai_response_free(message: str, message_type: str) -> dict:
    """IA gratuite optimisée pour Vercel avec sources spécialisées"""
    try:
        # Détection de questions sur l'anglais
        message_lower = message.lower()
        is_english_query = any(word in message_lower for word in [
            "english", "anglais", "grammar", "grammaire anglaise", "literature", 
            "writing", "essay", "esl", "pronunciation", "vocabulary"
        ])
        
        english_category = None
        sources_to_add = []
        detected_lang = detect_language(message)
        
        if is_english_query:
            if any(word in message_lower for word in ["grammar", "grammaire", "tense", "verb", "syntax"]):
                english_category = "grammar"
            elif any(word in message_lower for word in ["literature", "poem", "novel", "shakespeare", "poetry"]):
                english_category = "literature"
            elif any(word in message_lower for word in ["writing", "essay", "academic", "research", "citation"]):
                english_category = "academic"
            elif any(word in message_lower for word in ["esl", "learning english", "vocabulary", "pronunciation"]):
                english_category = "esl"
            else:
                english_category = "grammar"  # Par défaut
            
            sources_to_add = english_sources[english_category][:3]  # Top 3 sources
        
        # Réponses éducatives selon la langue détectée
        if detected_lang == "en" and is_english_query:
            # Réponse en anglais avec sources internationales
            response = f"For your question about '{message}', I recommend these reliable English sources:\n\n"
            for i, source in enumerate(sources_to_add, 1):
                response += f"{i}. {source}\n"
            
            response += "\n💡 Étienne's tip: Always cross-reference multiple academic sources for comprehensive understanding."
            
        elif "shakespeare" in message_lower or "literature" in message_lower:
            if detected_lang == "en":
                response = f"For Shakespeare and English literature, here are the best academic sources:\n\n1. CliffsNotes Literature Guides (cliffsnotes.com)\n2. SparkNotes Literature (sparknotes.com)\n3. Project Gutenberg for original texts (gutenberg.org)\n\nThese provide comprehensive analysis and are widely recognized in academic circles."
            else:
                response = f"Pour la littérature anglaise et Shakespeare, voici les meilleures sources académiques:\n\n1. CliffsNotes Literature Guides (cliffsnotes.com)\n2. SparkNotes Literature (sparknotes.com)\n3. Lecturia Academic Library (lecturia.com)\n\nCes sources offrent des analyses complètes reconnues académiquement."
