"""
Étienne API - Assistant IA Éducatif pour le Collège Champagneur
Version 2.0 - Optimisé pour Vercel
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timezone
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os
import logging
import uuid

# ============================================
# CONFIGURATION
# ============================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Étienne API - Assistant IA Éducatif",
    version="2.0",
    description="API pour l'assistant IA éducatif du Collège Champagneur"
)

# Configuration CORS - Autoriser toutes les origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Connection (optionnel)
mongo_url = os.environ.get('MONGO_URL')
db = None
if mongo_url:
    try:
        client = AsyncIOMotorClient(mongo_url)
        db = client[os.environ.get('DB_NAME', 'etienne_free')]
        logger.info("✅ MongoDB connecté")
    except Exception as e:
        logger.warning(f"⚠️ MongoDB non disponible: {e}")
        # ============================================
# MODÈLES PYDANTIC
# ============================================

class ChatRequest(BaseModel):
    message: str
    message_type: str
    session_id: Optional[str] = None

class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    message: str
    response: str
    message_type: str
    trust_score: Optional[float] = None
    sources: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class TextAnalysisRequest(BaseModel):
    text: str
    analysis_type: str = "complete"

class DocumentRequest(BaseModel):
    content: str
    title: str = "Document Étienne"
    format: str = "pdf"
    # ============================================
# SOURCES ACADÉMIQUES SPÉCIALISÉES
# ============================================

ENGLISH_SOURCES = {
    "grammar": [
        "Oxford English Grammar (oxford.com)",
        "Cambridge Grammar (cambridge.org)", 
        "Grammarly Blog (grammarly.com/blog)",
        "BBC Learning English (bbc.co.uk/learningenglish)",
        "British Council (learnenglish.britishcouncil.org)",
        "Merriam-Webster Dictionary (merriam-webster.com)"
    ],
    "literature": [
        "CliffsNotes Literature Guides (cliffsnotes.com)",
        "SparkNotes Literature (sparknotes.com)",
        "Lecturia Academic Library (lecturia.com)",
        "Project Gutenberg (gutenberg.org)",
        "Poetry Foundation (poetryfoundation.org)",
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

QUEBEC_SOURCES = [
    "Gouvernement du Québec (quebec.ca)",
    "MEES - Ministère de l'Éducation (education.gouv.qc.ca)",
    "Allô Prof (alloprof.qc.ca)",
    "BAnQ - Bibliothèque nationale (banq.qc.ca)",
    "Universités québécoises (UdeM, McGill, Laval, UQAM)"
]
# ============================================
# FONCTIONS UTILITAIRES - DÉTECTION
# ============================================

def detect_language(text: str) -> str:
    """Détecte la langue du message (français ou anglais)"""
    english_words = [
        "the", "and", "to", "of", "a", "in", "is", "it", "you", "that", 
        "he", "was", "for", "on", "are", "as", "with", "help", "me", 
        "can", "could", "would", "should", "what", "how", "where", 
        "when", "why", "grammar", "writing", "english"
    ]
    
    french_words = [
        "le", "de", "et", "à", "un", "il", "être", "en", "avoir", 
        "que", "pour", "dans", "ce", "son", "une", "sur", "aide", 
        "moi", "peux", "pourrais", "voudrais", "devrais", "quoi", 
        "comment", "où", "quand", "pourquoi", "grammaire", "français"
    ]
    
    text_lower = text.lower()
    words = text_lower.split()
    
    english_count = sum(1 for word in words if word in english_words)
    french_count = sum(1 for word in words if word in french_words)
    
    return "en" if english_count > french_count and english_count > 0 else "fr"
    def detect_ai_content(text: str) -> dict:
        """
    Détecte si un texte a été généré par IA
    Analyse les patterns, structure et vocabulaire
    """
    try:
        # Indicateurs typiques d'IA
        ai_indicators = [
            "as an ai", "i'm an ai", "as a language model", 
            "i don't have personal", "i cannot", "i can't provide", 
            "it's important to note", "however", "furthermore", 
            "moreover", "in conclusion", "to summarize", "in summary",
            "it is worth noting", "it should be noted"
        ]
        
        text_lower = text.lower()
        ai_score = 0.0
        detected_patterns = []
        
        # Vérification des patterns IA
        for indicator in ai_indicators:
            if indicator in text_lower:
                ai_score += 0.2
                detected_patterns.append(indicator)
        
        # Analyse de la structure (phrases uniformes)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) > 3:
            avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_length > 15:  # Phrases longues et complexes
                ai_score += 0.15
                detected_patterns.append("phrase_complexity")
        
        # Uniformité excessive (répétition de structure)
        if len(sentences) > 2:
            sentence_starts = [s.split()[0] if s.split() else "" for s in sentences]
            if len(set(sentence_starts)) < len(sentences) * 0.5:
                ai_score += 0.1
                detected_patterns.append("repetitive_structure")
        
        # Score final
        ai_probability = min(ai_score, 0.99)
        
        return {
            "ai_probability": round(ai_probability, 2),
            "is_likely_ai": ai_probability > 0.5,
            "confidence": "High" if ai_probability > 0.7 else "Medium" if ai_probability > 0.3 else "Low",
            "detected_patterns": detected_patterns[:5],
            "analysis": f"{'Texte probablement généré par IA' if ai_probability > 0.5 else 'Texte semble authentique'}"
        }
        
    except Exception as e:
        logger.error(f"Erreur détection IA: {e}")
        return {
            "ai_probability": 0.0,
            "is_likely_ai": False,
            "confidence": "Error",
            "detected_patterns": [],
            "error": str(e)
        }
        def check_plagiarism(text: str) -> dict:
    """
    Vérificateur de plagiat basique
    Analyse les phrases académiques communes et diversité du vocabulaire
    """
    try:
        # Phrases académiques communes (potentiellement plagiées)
        common_phrases = [
            "according to the study", "research shows that", 
            "studies have shown", "it has been proven that", 
            "experts agree that", "the data suggests", "furthermore", 
            "in addition", "however", "therefore", "consequently",
            "selon l'étude", "les recherches montrent", 
            "il a été prouvé", "les experts s'accordent"
        ]
        
        text_lower = text.lower()
        plagiarism_score = 0.0
        found_phrases = []
        
        # Vérification des phrases suspectes
        for phrase in common_phrases:
            if phrase in text_lower:
                plagiarism_score += 0.1
                found_phrases.append(phrase)
        
        # Analyse de diversité du vocabulaire
        words = text_lower.split()
        unique_words = set(words)
        vocabulary_diversity = len(unique_words) / len(words) if words else 0
        
        # Faible diversité = risque de plagiat
        if vocabulary_diversity < 0.6:
            plagiarism_score += 0.15
        
        # Score final
        plagiarism_risk = min(plagiarism_score, 0.99)
        
        return {
            "plagiarism_risk": round(plagiarism_risk, 2),
            "is_suspicious": plagiarism_risk > 0.4,
            "vocabulary_diversity": round(vocabulary_diversity, 2),
            "risk_level": "High" if plagiarism_risk > 0.6 else "Medium" if plagiarism_risk > 0.3 else "Low",
            "found_phrases": found_phrases[:3],
            "suspicious_phrases": found_phrases[:3],  # Alias pour compatibilité frontend
            "recommendation": "Vérifiez l'originalité avec des sources académiques" if plagiarism_risk > 0.4 else "Contenu semble original"
        }
        
    except Exception as e:
        logger.error(f"Erreur vérification plagiat: {e}")
        return {
            "plagiarism_risk": 0.0,
            "is_suspicious": False,
            "vocabulary_diversity": 0.0,
            "risk_level": "Error",
            "found_phrases": [],
            "suspicious_phrases": [],
            "error": str(e)
        }
        # ============================================
# FONCTION IA - GÉNÉRATION DE RÉPONSES
# ============================================

async def get_ai_response(message: str, message_type: str) -> dict:
    """
    Génère une réponse IA adaptée à la langue et au type de question
    Intègre sources québécoises et internationales
    """
    try:
        # Détection de la langue
        detected_lang = detect_language(message)
        message_lower = message.lower()
        
        # Détection de questions sur l'anglais
        is_english_query = any(word in message_lower for word in [
            "english", "anglais", "grammar", "grammaire anglaise", 
            "literature", "writing", "essay", "esl", "pronunciation", 
            "vocabulary", "shakespeare", "poem", "novel"
        ])
        
        # Catégorisation des questions anglaises
        english_category = None
        sources_to_add = []
        
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
            
            sources_to_add = ENGLISH_SOURCES[english_category][:3]
        
        # Génération de réponse selon la langue et le type
        response = ""
        
        if is_english_query and detected_lang == "en":
            # Réponse en anglais avec sources internationales
            response = f"For your question about {english_category if english_category else 'English'}, here are reliable academic sources:\n\n"
            for i, source in enumerate(sources_to_add, 1):
                response += f"{i}. {source}\n"
            response += "\n💡 Étienne's Tip: Always cross-reference multiple sources for comprehensive understanding and cite properly in academic work."
            
        elif is_english_query and detected_lang == "fr":
            # Réponse en français avec sources internationales
            response = f"Pour votre question sur l'anglais ({english_category if english_category else 'général'}), voici des sources académiques fiables:\n\n"
            for i, source in enumerate(sources_to_add, 1):
                response += f"{i}. {source}\n"
            response += "\n💡 Conseil d'Étienne: Croisez toujours plusieurs sources pour une compréhension complète et citez-les dans vos travaux académiques."
            
        else:
            # Question québécoise normale
            if detected_lang == "fr":
                response = f"Bonjour ! Je suis Étienne, votre assistant IA du Collège Champagneur.\n\n"
                response += f"Concernant votre question sur '{message[:50]}...', voici des ressources éducatives québécoises fiables:\n\n"
                for i, source in enumerate(QUEBEC_SOURCES[:3], 1):
                    response += f"{i}. {source}\n"
                response += "\n💡 Conseil d'Étienne: Privilégiez les sources .gouv.qc.ca et .edu pour vos recherches académiques."
            else:
                response = f"Hello! I'm Étienne, your AI assistant from Collège Champagneur.\n\n"
                response += f"For your question, I recommend these Quebec educational resources:\n\n"
                for i, source in enumerate(QUEBEC_SOURCES[:3], 1):
                    response += f"{i}. {source}\n"
                response += "\n💡 Étienne's Tip: Prioritize .gouv.qc.ca and .edu sources for academic research."
        
        # Adaptation selon le type de demande
        if message_type == "je_veux":
            response += "\n\n📚 N'hésitez pas à me poser des questions plus spécifiques pour approfondir!"
        elif message_type == "sources_fiables":
            response += "\n\n✅ Ces sources sont reconnues et citables dans vos travaux académiques."
        elif message_type == "activites":
            response += "\n\n🎯 Je peux vous aider à créer des exercices ou activités pédagogiques sur ce sujet."
        
        # Sources et trust score
        final_sources = sources_to_add if is_english_query else QUEBEC_SOURCES[:3]
        trust_score = 0.95 if (message_type == "sources_fiables" or is_english_query) else 0.85
        
        return {
            "response": response,
            "trust_score": trust_score,
            "sources": final_sources,
            "detected_language": detected_lang,
            "can_download": len(response) > 50
        }
        
    except Exception as e:
        logger.error(f"Erreur génération réponse IA: {e}")
        return {
            "response": "Étienne rencontre un problème temporaire. Consultez les ressources éducatives sur education.gouv.qc.ca",
            "trust_score": None,
            "sources": [],
            "detected_language": "fr"
        }
        # ============================================
# GÉNÉRATION DE DOCUMENTS
# ============================================

def generate_pdf_simple(title: str, content: str) -> BytesIO:
    """Génère un PDF simple avec branding Étienne"""
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        story = []
        
        # Titre
        story.append(Paragraph(title, styles['Title']))
        story.append(Spacer(1, 20))
        
        # Contenu - Diviser en paragraphes
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            if para.strip():
                # Remplacer les retours à la ligne simples par <br/>
                para_clean = para.replace('\n', '<br/>')
                p = Paragraph(para_clean, styles['Normal'])
                story.append(p)
                story.append(Spacer(1, 12))
        
        # Footer avec branding
        story.append(Spacer(1, 30))
        footer_style = styles['Normal']
        footer_text = f"<i>Généré par Étienne - Assistant IA du Collège Champagneur<br/>{datetime.now().strftime('%d/%m/%Y à %H:%M')}</i>"
        footer = Paragraph(footer_text, footer_style)
        story.append(footer)
        
        # Construction du PDF
        doc.build(story)
        buffer.seek(0)
        
        logger.info(f"✅ PDF généré: {title}")
        return buffer
        
    except Exception as e:
        logger.error(f"❌ Erreur génération PDF: {e}")
        raise
        # ============================================
# ENDPOINTS API
# ============================================

@app.get("/")
async def root_redirect():
    """Redirection vers /api"""
    return {"message": "Bienvenue sur Étienne API. Utilisez /api pour plus d'informations."}

@app.get("/api")
async def api_root():
    """Endpoint racine de l'API - Informations complètes"""
    return {
        "message": "Étienne API - Assistant IA Éducatif",
        "status": "active",
        "platform": "vercel",
        "backend": "working",
        "version": "2.0",
        "college": "Collège Champagneur",
        "features": [
            "chat",
            "sources_anglaises",
            "detection_ia",
            "verification_plagiat",
            "analyse_complete",
            "generation_documents"
        ],
        "endpoints": {
            "health": "GET /api/health",
            "chat": "POST /api/chat",
            "subjects": "GET /api/subjects",
            "analyze": "POST /api/analyze-text",
            "detect_ai": "POST /api/detect-ai",
            "plagiarism": "POST /api/check-plagiarism",
            "document": "POST /api/generate-document"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "platform": "vercel",
        "assistant": "Étienne",
        "version": "2.0",
        "database": "connected" if db else "not_required"
    }
    @app.post("/api/chat", response_model=ChatMessage)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint principal de chat avec Étienne
    Supporte FR/EN avec sources québécoises et internationales
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        logger.info(f"Chat request: {request.message[:50]}... | Type: {request.message_type}")
        
        # Génération de réponse IA
        ai_result = await get_ai_response(request.message, request.message_type)
        
        # Création du message
        chat_message = ChatMessage(
            session_id=session_id,
            message=request.message,
            response=ai_result["response"],
            message_type=request.message_type,
            trust_score=ai_result["trust_score"],
            sources=ai_result["sources"]
        )
        
        # Sauvegarde en DB (si disponible)
        if db:
            try:
                await db.chat_messages.insert_one(chat_message.dict())
                logger.info(f"✅ Message sauvegardé en DB")
            except Exception as e:
                logger.warning(f"⚠️ Sauvegarde DB échouée: {e}")
        
        return chat_message
        
    except Exception as e:
        logger.error(f"❌ Erreur chat: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement du chat: {str(e)}")
        @app.get("/api/subjects")
async def get_subjects():
    """Matières du système éducatif québécois"""
    return {
        "langues": {
            "name": "Langues",
            "subjects": ["Français", "Anglais", "Espagnol"]
        },
        "sciences": {
            "name": "Sciences & Mathématiques",
            "subjects": ["Mathématiques", "Physique", "Chimie", "Biologie", "Sciences et technologies"]
        },
        "sciences_humaines": {
            "name": "Sciences Humaines", 
            "subjects": ["Histoire", "Géographie", "Économie", "Monde contemporain"]
        },
        "formation_generale": {
            "name": "Formation Générale",
            "subjects": ["Éthique et culture religieuse", "Éducation physique"]
        },
        "arts": {
            "name": "Arts",
            "subjects": ["Arts plastiques", "Musique", "Art dramatique"]
        }
    }
    @app.post("/api/analyze-text")
async def analyze_text_complete(request: TextAnalysisRequest):
    """
    Analyse complète de texte: IA + Plagiat + Langue
    Retourne tous les indicateurs en un seul appel
    """
    try:
        logger.info(f"Analyse de texte: {len(request.text)} caractères")
        
        # Analyses parallèles
        ai_result = detect_ai_content(request.text)
        plagiarism_result = check_plagiarism(request.text)
        detected_language = detect_language(request.text)
        
        # Compilation des résultats
        return {
            "text_length": len(request.text),
            "word_count": len(request.text.split()),
            "detected_language": detected_language,
            "ai_detection": ai_result,
            "plagiarism_check": plagiarism_result,
            "overall_assessment": {
                "is_authentic": not ai_result["is_likely_ai"] and not plagiarism_result["is_suspicious"],
                "confidence_score": round((1 - ai_result["ai_probability"] + (1 - plagiarism_result["plagiarism_risk"])) / 2, 2),
                "recommendation": "Texte semble authentique" if not ai_result["is_likely_ai"] and not plagiarism_result["is_suspicious"] else "Vérification approfondie recommandée"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur analyse: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse: {str(e)}")

@app.post("/api/detect-ai")
async def detect_ai_endpoint(request: TextAnalysisRequest):
    """
    Endpoint spécifique pour détection de contenu IA
    Analyse patterns, structure et probabilité
    """
    try:
        logger.info(f"Détection IA: {len(request.text)} caractères")
        result = detect_ai_content(request.text)
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur détection IA: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur détection IA: {str(e)}")

@app.post("/api/check-plagiarism")
async def check_plagiarism_endpoint(request: TextAnalysisRequest):
    """
    Endpoint spécifique pour vérification de plagiat
    Analyse phrases académiques et diversité vocabulaire
    """
    try:
        logger.info(f"Vérification plagiat: {len(request.text)} caractères")
        result = check_plagiarism(request.text)
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur plagiat: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur vérification plagiat: {str(e)}")
        @app.post("/api/generate-document")
async def generate_document(request: DocumentRequest):
    """
    Génération de documents PDF avec branding Étienne
    Support: PDF uniquement pour le moment
    """
    try:
        if request.format != 'pdf':
            raise HTTPException(
                status_code=400, 
                detail="Seul le format PDF est supporté actuellement"
            )
        
        logger.info(f"Génération document: {request.title}")
        
        # Génération du PDF
        buffer = generate_pdf_simple(request.title, request.content)
        filename = f"{request.title.replace(' ', '_')}.pdf"
        
        return StreamingResponse(
            BytesIO(buffer.read()),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "application/pdf"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur génération document: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur génération: {str(e)}")

# ============================================
# HANDLER VERCEL
# ============================================

# Export de l'app pour Vercel
handler = app

# ============================================
# POINT D'ENTRÉE
# ============================================

if __name__ == "__main__":
    # Pour tests locaux uniquement
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
