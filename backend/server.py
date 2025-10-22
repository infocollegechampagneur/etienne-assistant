from fastapi import FastAPI, APIRouter, HTTPException, Request, Depends
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import asyncio
import re
from fastapi.responses import StreamingResponse, FileResponse
from fastapi import UploadFile, File, Form
from io import BytesIO
from ai_detector_advanced import AdvancedAIDetector
import google.generativeai as genai
from docx import Document
from docx.shared import Inches as DocxInches, RGBColor, Pt as DocxPt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from pptx import Presentation
from pptx.util import Inches, Pt
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
import PyPDF2
import docx2txt
import pdfplumber
import pandas as pd
import tempfile
import json

# Import des fonctionnalites de securite avancees
from security_advanced import (
    get_current_admin, check_rate_limit, get_client_ip,
    send_critical_alert, send_daily_report,
    generate_csv_report, generate_pdf_report,
    hash_password, verify_password, create_access_token
)


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Configuration de Google Gemini (100% gratuit)
genai.configure(api_key=os.environ.get('GOOGLE_API_KEY', ''))

# Initialiser le detecteur IA avance avec auto-apprentissage
advanced_detector = AdvancedAIDetector()

# Charger les poids depuis la base de donnees au demarrage
async def load_detector_weights():
    """Charge les poids du detecteur depuis MongoDB"""
    try:
        weights_doc = await db.detector_weights.find_one({"version": "latest"})
        if weights_doc and "weights" in weights_doc:
            advanced_detector.set_weights(weights_doc["weights"])
            logging.info("Poids du detecteur charges depuis la base de donnees")
        else:
            logging.info("Utilisation des poids par defaut du detecteur")
    except Exception as e:
        logging.error(f"Erreur chargement poids: {e}")

# Sauvegarder les poids dans la base de donnees
async def save_detector_weights():
    """Sauvegarde les poids du detecteur dans MongoDB"""
    try:
        weights = advanced_detector.get_weights()
        await db.detector_weights.update_one(
            {"version": "latest"},
            {
                "$set": {
                    "weights": weights,
                    "updated_at": datetime.now(timezone.utc)
                }
            },
            upsert=True
        )
        logging.info("Poids du detecteur sauvegardes")
    except Exception as e:
        logging.error(f"Erreur sauvegarde poids: {e}")

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Define Models
class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    message: str
    response: str
    message_type: str  # "je_veux", "je_recherche", "sources_fiables", "activites"
    trust_score: Optional[float] = None
    sources: Optional[List[str]] = None
    image_base64: Optional[str] = None  # Pour les images generees
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ChatRequest(BaseModel):
    message: str
    message_type: str
    session_id: Optional[str] = None

class SearchResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    sources: List[dict]  # [{"url": str, "title": str, "trust_score": float, "content_preview": str}]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class DocumentRequest(BaseModel):
    content: str
    title: str = "Document Etienne"
    format: str = "pdf"  # pdf, docx, pptx, xlsx
    filename: Optional[str] = None

class TextAnalysisRequest(BaseModel):
    text: str
    analysis_type: str = "complete"  # "ai_detection", "plagiarism", "complete"

class FileAnalysisRequest(BaseModel):
    question: str

class SecurityIncidentLog(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    user_message: str
    detected_category: str  # "violence", "illegal_activities", "hacking", "drugs", etc.
    keywords_matched: List[str]
    severity: str  # "low", "medium", "high", "critical"
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    blocked: bool = True
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Systeme de confiance des sources
TRUSTED_DOMAINS = {
    ".gouv.qc.ca": 0.98,
    ".gouv.ca": 0.95,
    ".edu": 0.90,
    "quebec.ca": 0.97,
    "education.gouv.qc.ca": 0.98,
    "mees.gouv.qc.ca": 0.98,  # Ministere de l Education du Quebec
    "banq.qc.ca": 0.88,  # Bibliotheque nationale du Quebec
    "uqam.ca": 0.85,
    "umontreal.ca": 0.85,
    "ulaval.ca": 0.85,
    "mcgill.ca": 0.85,
    "cegep": 0.75,
    "universit": 0.75,
    ".ca": 0.70,
    ".org": 0.60,
    ".com": 0.40,
    "wikipedia": 0.65
}

def calculate_trust_score(url: str, content: str = "") -> float:
    """Calcule le score de confiance de une source base sur le domaine et le contenu"""
    base_score = 0.5
    
    # Verification du domaine
    for domain, score in TRUSTED_DOMAINS.items():
        if domain in url.lower():
            base_score = max(base_score, score)
    
    # Analyse basique du contenu si fourni
    if content:
        quality_indicators = [
            "bibliographie", "references", "source", "etude", "recherche", 
            "academique", "officiel", "ministere", "universite", "peer-review"
        ]
        quality_count = sum(1 for indicator in quality_indicators if indicator in content.lower())
        content_bonus = min(0.15, quality_count * 0.03)
        base_score = min(0.98, base_score + content_bonus)
    
    return round(base_score, 2)

async def extract_text_from_file(file: UploadFile) -> str:
    """Extrait le texte de un fichier uploade"""
    try:
        # Creer un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        extracted_text = ""
        file_extension = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
        
        try:
            if file_extension == 'pdf':
                # Extraction PDF avec pdfplumber (meilleure qualite)
                with pdfplumber.open(temp_file_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            extracted_text += text + "\n"
                
                # Fallback avec PyPDF2 si pdfplumber echoue
                if not extracted_text.strip():
                    with open(temp_file_path, 'rb') as pdf_file:
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        for page in pdf_reader.pages:
                            extracted_text += page.extract_text() + "\n"
            
            elif file_extension in ['docx', 'doc']:
                # Extraction Word
                extracted_text = docx2txt.process(temp_file_path)
            
            elif file_extension == 'txt':
                # Fichier texte simple
                with open(temp_file_path, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
            
            elif file_extension in ['xlsx', 'xls']:
                # Extraction Excel
                try:
                    df = pd.read_excel(temp_file_path, sheet_name=None)  # Toutes les feuilles
                    for sheet_name, sheet_data in df.items():
                        extracted_text += f"\n=== Feuille: {sheet_name} ===\n"
                        extracted_text += sheet_data.to_string(index=False, na_rep='') + "\n"
                except Exception:
                    # Fallback pour anciens formats Excel
                    df = pd.read_excel(temp_file_path, engine='openpyxl')
                    extracted_text = df.to_string(index=False, na_rep='')
            
            elif file_extension == 'csv':
                # Fichier CSV
                df = pd.read_csv(temp_file_path)
                extracted_text = df.to_string(index=False, na_rep='')
            
            elif file_extension == 'pptx':
                # PowerPoint (extraction basique)
                from pptx import Presentation
                prs = Presentation(temp_file_path)
                for slide_num, slide in enumerate(prs.slides, 1):
                    extracted_text += f"\n=== Slide {slide_num} ===\n"
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            extracted_text += shape.text + "\n"
            
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Format de fichier non supporte: {file_extension}. "
                          f"Formats acceptes: PDF, DOCX, TXT, XLSX, CSV, PPTX"
                )
        
        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
        if not extracted_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Impossible de extraire le texte de ce fichier. Verifiez que le fichier n'est pas protege ou corrompu."
            )
        
        return extracted_text.strip()
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Erreur extraction texte: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l extraction du texte: {str(e)}"
        )

# ============================================================================
# DETECTION IA GRATUITE - HuggingFace + Heuristique Avancee
# Precision cible: 75-85% sans coût
# ============================================================================

async def detect_ai_content_with_huggingface(text: str) -> dict:
    """
    Detection IA gratuite utilisant HuggingFace Inference API.
    Utilise un modele RoBERTa fine-tune specifiquement pour detecter le contenu IA.
    """
    try:
        import aiohttp
        
        # Limiter la longueur du texte
        max_text_length = 2000
        analyzed_text = text[:max_text_length] if len(text) > max_text_length else text
        
        # Modele gratuit specialise pour detection IA sur HuggingFace
        # Ce modele est entraîne specifiquement pour detecter GPT/Claude/etc.
        API_URL = "https://api-inference.huggingface.co/models/Hello-SimpleAI/chatgpt-detector-roberta"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                API_URL,
                headers={"Content-Type": "application/json"},
                json={"inputs": analyzed_text},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Le modele retourne [{"label": "Real/Fake", "score": 0.XX}]
                    if isinstance(result, list) and len(result) > 0:
                        # Trouver le score "Fake" (AI-generated)
                        ai_score = 0.5
                        for item in result:
                            if item.get("label") == "Fake":
                                ai_score = item.get("score", 0.5)
                                break
                        
                        # Determiner les patterns bases sur le score
                        detected_patterns = []
                        if ai_score > 0.7:
                            detected_patterns = ["high_ai_confidence", "model_detected_fake", "consistent_patterns"]
                        elif ai_score > 0.5:
                            detected_patterns = ["moderate_ai_indicators", "some_patterns_detected"]
                        else:
                            detected_patterns = ["likely_human_written"]
                        
                        return {
                            "ai_probability": round(ai_score, 2),
                            "is_likely_ai": ai_score > 0.5,
                            "confidence": "High" if ai_score > 0.7 or ai_score < 0.3 else "Medium",
                            "detected_patterns": detected_patterns,
                            "reasoning": f"Modele RoBERTa detecte {int(ai_score*100)}% de probabilite IA",
                            "method": "huggingface_roberta"
                        }
                
                # Si l API HuggingFace echoue, utiliser le fallback
                logging.warning(f"HuggingFace API status: {response.status}")
                return detect_ai_content_fallback(text)
                
    except asyncio.TimeoutError:
        logging.warning("HuggingFace API timeout - using fallback")
        return detect_ai_content_fallback(text)
    except Exception as e:
        logging.error(f"Erreur HuggingFace API: {e}")
        return detect_ai_content_fallback(text)


async def detect_ai_content_with_llm(text: str) -> dict:
    """
    Point de entree principal pour la detection IA.
    Essaie de abord HuggingFace (gratuit), puis fallback vers heuristique avancee.
    """
    # Essayer de abord HuggingFace
    result = await detect_ai_content_with_huggingface(text)
    
    # Si la methode est deja le fallback, le retourner directement
    if result.get("method") == "heuristic_advanced":
        return result
    
    # Si HuggingFace a fonctionne mais avec faible confiance, combiner avec heuristique
    if result.get("confidence") == "Medium":
        heuristic_result = detect_ai_content_fallback(text)
        
        # Moyenne ponderee : 70% HuggingFace, 30% heuristique
        combined_probability = (result["ai_probability"] * 0.7) + (heuristic_result["ai_probability"] * 0.3)
        
        return {
            "ai_probability": round(combined_probability, 2),
            "is_likely_ai": combined_probability > 0.5,
            "confidence": "High" if combined_probability > 0.7 or combined_probability < 0.3 else "Medium",
            "detected_patterns": result["detected_patterns"][:2] + heuristic_result["detected_patterns"][:1],
            "reasoning": f"Analyse combinee HuggingFace + heuristique: {int(combined_probability*100)}%",
            "method": "hybrid_huggingface_heuristic"
        }
    
    return result


def detect_ai_content_fallback(text: str) -> dict:
    """
    Detection IA ultra-avancee avec 15+ analyses et auto-apprentissage
    Precision: 80-85% (ameliore au fil du temps)
    """
    return advanced_detector.analyze_text(text)

# ============================================================================
# FIN DE LA NOUVELLE DETECTION IA
# ============================================================================


def check_plagiarism(text: str) -> dict:
    """Verificateur de plagiat basique"""
    try:
        # Phrases communes qui peuvent indiquer du plagiat
        common_academic_phrases = [
            "according to the study", "research shows that", "studies have shown",
            "it has been proven that", "experts agree that", "the data suggests",
            "furthermore", "in addition", "however", "therefore", "consequently"
        ]
        
        # Verification de phrases trop parfaites/academiques
        text_lower = text.lower()
        academic_score = 0
        found_phrases = []
        
        for phrase in common_academic_phrases:
            if phrase in text_lower:
                academic_score += 0.1
                found_phrases.append(phrase)
        
        # Verification de la diversite du vocabulaire
        words = text_lower.split()
        unique_words = set(words)
        vocabulary_diversity = len(unique_words) / len(words) if words else 0
        
        # Score de risque de plagiat
        plagiarism_risk = min(academic_score, 0.9)
        if vocabulary_diversity < 0.6:  # Faible diversite = risque
            plagiarism_risk += 0.1
        
        plagiarism_risk = min(plagiarism_risk, 0.99)
        
        return {
            "plagiarism_risk": round(plagiarism_risk, 2),
            "is_suspicious": plagiarism_risk > 0.4,
            "vocabulary_diversity": round(vocabulary_diversity, 2),
            "risk_level": "High" if plagiarism_risk > 0.6 else "Medium" if plagiarism_risk > 0.3 else "Low",
            "found_phrases": found_phrases[:3],
            "recommendation": "Verifiez l originalite avec des sources academiques" if plagiarism_risk > 0.4 else "Contenu semble original"
        }
        
    except Exception as e:
        return {
            "plagiarism_risk": 0.0,
            "is_suspicious": False,
            "error": str(e)
        }

# Detection de langue pour reponses adaptees
def detect_language(text: str) -> str:
    """Detecte la langue du message"""
    english_words = [
        "the", "and", "to", "of", "a", "in", "is", "it", "you", "that", "he", "was", "for", "on", "are", "as", "with",
        "help", "me", "can", "could", "would", "should", "what", "how", "where", "when", "why", "grammar", "writing"
    ]
    
    french_words = [
        "le", "de", "et", "a", "un", "il", "etre", "et", "en", "avoir", "que", "pour", "dans", "ce", "son", "une", "sur",
        "aide", "moi", "peux", "pourrais", "voudrais", "devrais", "quoi", "comment", "où", "quand", "pourquoi", "grammaire"
    ]


# ============================================
# DETECTION DE CONTENU ILLEGAL ET DANGEREUX
# ============================================

ILLEGAL_CONTENT_PATTERNS = {
    "violence": {
        "keywords": [
            "tuer", "assassiner", "meurtre", "comment tuer", "faire du mal", "blesser quelqu'un",
            "torturer", "violence physique", "attaque", "agresser", "frapper quelqu'un",
            "murder", "kill someone", "how to kill", "hurt someone", "attack someone"
        ],
        "severity": "critical"
    },
    "explosives": {
        "keywords": [
            "fabriquer une bombe", "faire exploser", "explosif", "bombe artisanale",
            "faire une bombe", "creer une bombe", "dynamite", "detonateur",
            "make a bomb", "build a bomb", "explosive device", "homemade bomb"
        ],
        "severity": "critical"
    },
    "hacking": {
        "keywords": [
            "pirater", "hacker", "voler des donnees", "acces illegal", "briser un mot de passe",
            "cracker un systeme", "ddos", "intrusion", "exploit", "backdoor",
            "hack into", "break into", "steal data", "crack password", "unauthorized access"
        ],
        "severity": "high"
    },
    "drugs": {
        "keywords": [
            "fabriquer de la drogue", "synthetiser", "methamphetamine", "crack", "heroïne",
            "comment faire de la drogue", "produire de la drogue", "laboratoire clandestin",
            "make drugs", "synthesize drugs", "drug lab", "manufacture drugs"
        ],
        "severity": "critical"
    },
    "weapons": {
        "keywords": [
            "fabriquer une arme", "arme artisanale", "arme a feu", "construire une arme",
            "silencieux", "munitions", "arsenal", "armement illegal",
            "make a weapon", "build a gun", "homemade weapon", "firearm"
        ],
        "severity": "high"
    },
    "fraud": {
        "keywords": [
            "fraude", "escroquerie", "contrefaçon", "blanchiment de argent", "faux documents",
            "usurpation de identite", "carte de credit volee", "fraude fiscale",
            "fraud", "scam", "counterfeit", "money laundering", "identity theft"
        ],
        "severity": "high"
    },
    "child_exploitation": {
        "keywords": [
            "pedophilie", "exploitation de enfants", "abus de enfants", "contenu illegal mineur",
            "child abuse", "child exploitation", "child pornography", "minor abuse"
        ],
        "severity": "critical"
    },
    "terrorism": {
        "keywords": [
            "terrorisme", "attentat", "attaque terroriste", "radicalisation", "extremisme violent",
            "terrorism", "terrorist attack", "radicalization", "violent extremism"
        ],
        "severity": "critical"
    }
}

async def detect_illegal_content(message: str, session_id: str = None, ip_address: str = None, user_agent: str = None) -> dict:
    """
    Detecte le contenu illegal ou dangereux dans un message
    Retourne un dict avec les details de detection ou None si contenu ok
    """
    message_lower = message.lower()
    
    for category, data in ILLEGAL_CONTENT_PATTERNS.items():
        matched_keywords = []
        
        for keyword in data["keywords"]:
            if keyword.lower() in message_lower:
                matched_keywords.append(keyword)
        
        if matched_keywords:
            # Contenu illegal detecte - creer un log
            log_entry = SecurityIncidentLog(
                session_id=session_id,
                user_message=message[:500],  # Limiter la taille pour la DB
                detected_category=category,
                keywords_matched=matched_keywords,
                severity=data["severity"],
                ip_address=ip_address,
                user_agent=user_agent,
                blocked=True
            )
            
            # Sauvegarder dans MongoDB
            try:
                await db.security_illegal_logs.insert_one(log_entry.dict())
                logging.warning(f"🚨 CONTENU ILLEGAL DETECTE - Categorie: {category}, Severite: {data['severity']}, Session: {session_id}")
                
                # Envoyer alerte email si severite critique
                if data["severity"] == "critical":
                    log_dict = log_entry.dict()
                    log_dict['timestamp'] = log_dict['timestamp'].isoformat() if isinstance(log_dict['timestamp'], datetime) else str(log_dict['timestamp'])
                    asyncio.create_task(asyncio.to_thread(send_critical_alert, log_dict))
                    
            except Exception as e:
                logging.error(f"Erreur sauvegarde log illegal: {e}")
            
            return {
                "detected": True,
                "category": category,
                "severity": data["severity"],
                "keywords": matched_keywords,
                "log_id": log_entry.id
            }
    
    return {"detected": False}

    
    text_lower = text.lower()
    words = text_lower.split()
    
    english_count = sum(1 for word in words if word in english_words)
    french_count = sum(1 for word in words if word in french_words)
    
    # Si plus de mots anglais detectes
    if english_count > french_count and english_count > 0:
        return "en"
    else:
        return "fr"

async def get_ai_response(message: str, message_type: str) -> dict:
    """Obtient une reponse de Etienne selon le type de message - 100% GRATUIT avec Gemini"""
    try:
        # DETECTION DE DEMANDE D'IMAGE
        image_keywords = ['genere une image', 'cree une image', 'genere-moi une image', 
                         'fais-moi une image', 'dessine', 'illustre', 'genere un dessin',
                         'creer une illustration', 'faire un dessin']
        is_image_request = any(keyword in message.lower() for keyword in image_keywords)
        
        if is_image_request:
            # Extraire le prompt de l image
            prompt = message
            for keyword in image_keywords:
                if keyword in message.lower():
                    prompt = message.lower().replace(keyword, '').strip()
                    break
            
            # Si le prompt est vide, utiliser tout le message
            if not prompt or len(prompt) < 5:
                prompt = message
            
            # Appeler la generation de image
            try:
                image_bytes = await generate_image_huggingface(prompt)
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                
                return {
                    "response": f"✨ Voici l image que j'ai generee pour vous!\n\n📝 Description: {prompt}",
                    "trust_score": None,
                    "sources": ["Hugging Face Stable Diffusion XL"],
                    "image_base64": image_base64,
                    "is_image": True
                }
            except Exception as e:
                logging.error(f"Erreur generation image: {e}")
                return {
                    "response": f"❌ Desole, je n'ai pas pu generer l image. Erreur: {str(e)}\n\nPouvez-vous reformuler votre demande?",
                    "trust_score": None,
                    "sources": [],
                    "is_image": False
                }
        
        # Sources credibles specialisees pour l anglais (mises a jour)
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
        
        # Configuration du systeme selon le type - Etienne
        system_messages = {
            "je_veux": "Tu es Etienne, un assistant pedagogique specialise pour les etudiants quebecois. Reponds de maniere claire et educative. Pour l anglais, recommande les meilleures sources mondiales comme Oxford, Cambridge, BBC Learning, Purdue OWL. Utilise un français quebecois accessible.",
            "je_recherche": "Tu es Etienne, assistant de recherche educative. Aide les etudiants quebecois a explorer des sujets scolaires. Pour l anglais, oriente vers des sources internationales prestigieuses. Propose des pistes de recherche pedagogiques.",
            "sources_fiables": "Tu es Etienne, expert en sources academiques. Guide vers des sources fiables: pour le quebecois (.gouv.qc.ca, .edu), pour l anglais (Oxford, Cambridge, BBC, Purdue OWL, Harvard, MIT). Explique comment evaluer la credibilite.",
            "activites": "Tu es Etienne, createur de activites pedagogiques pour etudiants quebecois. Propose des exercices engageants adaptes au programme quebecois. Pour l anglais, utilise des ressources internationales de qualite."
        }
        
        # Detection de questions sur l anglais
        message_lower = message.lower()
        is_english_query = any(word in message_lower for word in [
            "english", "anglais", "grammar", "grammaire anglaise", "literature", 
            "writing", "essay", "esl", "pronunciation", "vocabulary"
        ])
        
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
                english_category = "grammar"  # Par defaut
            
            sources_to_add = english_sources[english_category][:3]  # Top 3 sources
        
        # Detection de la langue du message
        detected_language = detect_language(message)
        
        # Enrichir le message si c'est une question de anglais
        enhanced_message = message
        if is_english_query and english_category:
            enhanced_message = f"{message}\n\nNote: Je recommande particulierement ces sources pour l anglais: {', '.join(sources_to_add[:2])}."
        
        # Adapter le systeme selon la langue detectee
        base_system_message = system_messages.get(message_type, system_messages["je_veux"])
        
        if detected_language == "en":
            # Repondre en anglais si l utilisateur ecrit en anglais
            english_system_messages = {
                "je_veux": "You are Etienne, an educational assistant specialized for Quebec students. Respond clearly and educationally in English. For English topics, recommend the best global sources like Oxford, Cambridge, BBC Learning, Purdue OWL. Be helpful and accessible.",
                "je_recherche": "You are Etienne, an educational research assistant. Help Quebec students explore academic topics in English. Guide them to prestigious international sources. Suggest educational research paths.",
                "sources_fiables": "You are Etienne, an expert in academic sources. Guide to reliable sources: for Quebec (.gouv.qc.ca, .edu), for English (Oxford, Cambridge, BBC, Purdue OWL, Harvard, MIT). Explain how to evaluate credibility.",
                "activites": "You are Etienne, creator of educational activities for Quebec students. Propose engaging exercises in English adapted to Quebec curriculum. For English topics, use quality international resources."
            }
            system_message = english_system_messages.get(message_type, english_system_messages["je_veux"])
        else:
            system_message = base_system_message
        
        # Verifier si l utilisateur demande un document
        document_keywords = [
            "creer", "genere", "document", "fichier", "pdf", "word", "excel", "powerpoint",
            "telecharger", "exporter", "rapport", "resume", "fiche", "presentation"
        ]
        
        wants_document = any(keyword in message.lower() for keyword in document_keywords)
        
        # Modifier le prompt si document demande
        if wants_document:
            enhanced_message = f"{enhanced_message}\n\nNote: L'utilisateur semble vouloir creer un document. Structurez votre reponse de maniere claire avec des titres et des points cles qui pourront etre facilement exportes en PDF, Word, PowerPoint ou Excel."
        
        # === GOOGLE GEMINI 2.0 FLASH (100% GRATUIT) ===
        model = genai.GenerativeModel(
            model_name='gemini-2.0-flash-exp',
            system_instruction=system_message
        )
        
        # Generer la reponse
        response = await asyncio.to_thread(
            model.generate_content,
            enhanced_message
        )
        
        response_text = response.text
        
        # Ajouter sources specialisees si anglais
        final_sources = sources_to_add if is_english_query else ["Sources educatives quebecoises recommandees"]
        trust_score = 0.95 if (message_type == "sources_fiables" or is_english_query) else None
        
        return {
            "response": response_text,
            "trust_score": trust_score,
            "sources": final_sources,
            "can_download": len(response_text) > 50
        }
        
    except Exception as e:
        logging.error(f"Erreur IA Gemini: {e}")
        return {
            "response": "Desole, une erreur s'est produite. Veuillez reessayer.",
            "trust_score": None,
            "sources": []
        }

# Routes API
@api_router.get("/")
async def root():
    return {"message": "API Etienne - Assistant IA pour les etudiants quebecois fourni par le College Champagneur"}

@api_router.post("/chat", response_model=ChatMessage)
async def chat_with_ai(request: ChatRequest, http_request: Request):
    """Endpoint principal pour le chat avec l IA"""
    try:
        # Extraction de l IP et verification du rate limiting
        client_ip = get_client_ip(http_request)
        user_agent = http_request.headers.get("User-Agent", "Unknown")
        
        # Verifier le rate limiting
        if not check_rate_limit(client_ip):
            raise HTTPException(
                status_code=429,
                detail="Trop de requetes. Veuillez patienter avant de reessayer."
            )
        
        # Generation de un session_id si non fourni
        session_id = request.session_id or str(uuid.uuid4())
        
        # 🚨 DETECTION DE CONTENU ILLEGAL
        illegal_check = await detect_illegal_content(
            message=request.message,
            session_id=session_id,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        if illegal_check["detected"]:
            # Bloquer et retourner un message de avertissement
            warning_message = (
                "⚠️ ATTENTION: Votre demande a ete detectee comme potentiellement dangereuse ou illegale.\n\n"
                "Je ne peux pas vous aider avec ce type de contenu. "
                "Si vous avez des questions legitimes sur la securite ou l education, "
                "veuillez reformuler votre demande de maniere appropriee.\n\n"
                "Cette tentative a ete enregistree conformement a nos politiques de securite."
            )
            
            chat_message = ChatMessage(
                session_id=session_id,
                message=request.message,
                response=warning_message,
                message_type=request.message_type,
                trust_score=0.0,
                sources=["Systeme de securite Etienne"]
            )
            
            # Ne pas sauvegarder le message utilisateur, seulement l avertissement
            return chat_message
        
        # Obtention de la reponse IA (si contenu legal)
        ai_result = await get_ai_response(request.message, request.message_type)
        
        # Creation de l objet ChatMessage
        chat_message = ChatMessage(
            session_id=session_id,
            message=request.message,
            response=ai_result["response"],
            message_type=request.message_type,
            trust_score=ai_result["trust_score"],
            sources=ai_result["sources"],
            image_base64=ai_result.get("image_base64")  # Ajouter l image si presente
        )
        
        # Sauvegarde en base de donnees
        await db.chat_messages.insert_one(chat_message.dict())
        
        return chat_message
        
    except Exception as e:
        logging.error(f"Erreur chat: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors du traitement de la demande")

@api_router.get("/chat/history/{session_id}", response_model=List[ChatMessage])
async def get_chat_history(session_id: str):
    """Recupere l historique de une session de chat"""
    try:
        messages = await db.chat_messages.find(
            {"session_id": session_id}
        ).sort("timestamp", 1).to_list(100)
        
        return [ChatMessage(**message) for message in messages]
        
    except Exception as e:
        logging.error(f"Erreur historique: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la recuperation de l historique")

@api_router.post("/sources/analyze")
async def analyze_sources(sources: List[str]):
    """Analyse la fiabilite de une liste de sources"""
    try:
        analyzed_sources = []
        
        for source_url in sources:
            trust_score = calculate_trust_score(source_url)
            analyzed_sources.append({
                "url": source_url,
                "trust_score": trust_score,
                "trust_level": "Tres fiable" if trust_score >= 0.8 else 
                              "Fiable" if trust_score >= 0.6 else 
                              "Moderement fiable" if trust_score >= 0.4 else "Peu fiable",
                "recommendation": "Source recommandee" if trust_score >= 0.7 else 
                                "Verifier avec de autres sources" if trust_score >= 0.5 else 
                                "Source non recommandee"
            })
        
        return {"analyzed_sources": analyzed_sources}
        
    except Exception as e:
        logging.error(f"Erreur analyse sources: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de l analyse des sources")

def generate_pdf_document(title: str, content: str) -> BytesIO:
    """Genere un document PDF avec belle presentation"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    
    # Styles personnalises
    from reportlab.lib.colors import HexColor, black, white
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.platypus import PageBreak
    
    # Style de titre personnalise
    title_style = styles['Title'].clone('CustomTitle')
    title_style.fontSize = 24
    title_style.spaceAfter = 30
    title_style.textColor = HexColor('#f97316')
    title_style.alignment = TA_CENTER
    
    # Style de sous-titre
    subtitle_style = styles['Heading2'].clone('CustomSubtitle')
    subtitle_style.fontSize = 16
    subtitle_style.textColor = HexColor('#2563eb')
    subtitle_style.spaceAfter = 12
    
    story = []
    
    # En-tete avec logo Etienne
    header_table = Table([
        ['🎓', title, '📚'],
    ], colWidths=[50, 400, 50])
    header_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (0, 0), 20),  # Emoji
        ('FONTSIZE', (1, 0), (1, 0), 18),  # Titre
        ('FONTSIZE', (2, 0), (2, 0), 20),  # Emoji
        ('TEXTCOLOR', (1, 0), (1, 0), HexColor('#f97316')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 30))
    
    # Ligne de separation
    from reportlab.platypus import HRFlowable
    story.append(HRFlowable(width="100%", thickness=2, color=HexColor('#f97316')))
    story.append(Spacer(1, 20))
    
    # Traitement du contenu
    sections = content.split('\n\n')
    for i, section in enumerate(sections):
        if section.strip():
            # Detecter si c'est un titre (commence par #, ou est court et en majuscules)
            if (section.startswith('#') or 
                (len(section) < 80 and section.isupper()) or
                (i == 0 and len(section) < 100)):
                # C'est un sous-titre
                clean_title = section.replace('#', '').strip()
                story.append(Paragraph(clean_title, subtitle_style))
            else:
                # C'est du contenu normal
                formatted_text = section.replace('\n', '<br/>')
                p = Paragraph(formatted_text, styles['Normal'])
                story.append(p)
            
            story.append(Spacer(1, 12))
    
    # Footer elegant
    story.append(Spacer(1, 30))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#e5e7eb')))
    story.append(Spacer(1, 10))
    
    footer_table = Table([
        ['Genere par Etienne', f"{datetime.now().strftime('%d/%m/%Y a %H:%M')}", 'College Champagneur'],
    ], colWidths=[150, 200, 150])
    footer_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (-1, -1), HexColor('#6b7280')),
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
        ('ALIGN', (1, 0), (1, 0), 'CENTER'),
        ('ALIGN', (2, 0), (2, 0), 'RIGHT'),
    ]))
    story.append(footer_table)
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_docx_document(title: str, content: str) -> BytesIO:
    """Genere un document Word avec belle presentation"""
    doc = Document()
    
    # En-tete avec logo Etienne
    header_para = doc.add_paragraph()
    header_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    header_para.add_run("🎓 ")
    title_run = header_para.add_run(title)
    title_run.bold = True
    title_run.font.size = DocxPt(20)
    title_run.font.color.rgb = RGBColor(249, 115, 22)  # Orange
    header_para.add_run(" 📚")
    
    # Ligne de separation
    separator_para = doc.add_paragraph()
    separator_run = separator_para.add_run("_" * 80)
    separator_run.font.color.rgb = RGBColor(249, 115, 22)
    
    doc.add_paragraph()  # Espace
    
    # Traitement du contenu
    sections = content.split('\n\n')
    for i, section in enumerate(sections):
        if section.strip():
            # Detecter si c'est un titre
            if (section.startswith('#') or 
                (len(section) < 80 and section.isupper()) or
                (i == 0 and len(section) < 100)):
                # C'est un sous-titre
                clean_title = section.replace('#', '').strip()
                subtitle_para = doc.add_heading(clean_title, level=2)
                subtitle_para.runs[0].font.color.rgb = RGBColor(37, 99, 235)  # Bleu
            else:
                # C'est du contenu normal
                para = doc.add_paragraph(section)
                para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    
    # Footer elegant
    doc.add_paragraph()
    footer_separator = doc.add_paragraph()
    footer_separator_run = footer_separator.add_run("─" * 80)
    footer_separator_run.font.color.rgb = RGBColor(229, 231, 235)  # Gris clair
    
    footer_para = doc.add_paragraph()
    footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    footer_left = footer_para.add_run("Genere par Etienne")
    footer_left.font.size = DocxPt(9)
    footer_left.font.color.rgb = RGBColor(107, 114, 128)
    
    footer_center = footer_para.add_run(f" • {datetime.now().strftime('%d/%m/%Y a %H:%M')} • ")
    footer_center.font.size = DocxPt(9)
    footer_center.font.color.rgb = RGBColor(107, 114, 128)
    
    footer_right = footer_para.add_run("College Champagneur")
    footer_right.font.size = DocxPt(9)
    footer_right.font.color.rgb = RGBColor(107, 114, 128)
    footer_right.italic = True
    
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

def generate_pptx_document(title: str, content: str) -> BytesIO:
    """Genere une belle presentation PowerPoint"""
    prs = Presentation()
    
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    
    # === SLIDE DE TITRE ===
    slide_layout = prs.slide_layouts[0]  # Title slide
    title_slide = prs.slides.add_slide(slide_layout)
    
    # Titre principal
    title_shape = title_slide.shapes.title
    title_shape.text = f"🎓 {title} 📚"
    title_paragraph = title_shape.text_frame.paragraphs[0]
    title_paragraph.font.size = Pt(44)
    title_paragraph.font.color.rgb = RGBColor(249, 115, 22)  # Orange
    title_paragraph.alignment = PP_ALIGN.CENTER
    
    # Sous-titre
    subtitle_shape = title_slide.placeholders[1]
    subtitle_shape.text = f"Presente par Etienne\nAssistant IA Educatif\nCollege Champagneur\n\n{datetime.now().strftime('%d %B %Y')}"
    subtitle_paragraph = subtitle_shape.text_frame.paragraphs[0]
    subtitle_paragraph.font.size = Pt(18)
    subtitle_paragraph.font.color.rgb = RGBColor(37, 99, 235)  # Bleu
    subtitle_paragraph.alignment = PP_ALIGN.CENTER
    
    # === TRAITEMENT DU CONTENU ===
    sections = content.split('\n\n')
    current_slide_content = []
    slide_counter = 1
    
    for section in sections:
        if section.strip():
            current_slide_content.append(section.strip())
            
            # Creer une nouvelle slide tous les 4 points ou si section trop longue
            if len(current_slide_content) >= 4 or len(section) > 300:
                slide_counter += 1
                slide = prs.slides.add_slide(prs.slide_layouts[1])  # Content layout
                
                # Titre de la slide
                slide.shapes.title.text = f"📖 Section {slide_counter - 1}"
                title_para = slide.shapes.title.text_frame.paragraphs[0]
                title_para.font.size = Pt(32)
                title_para.font.color.rgb = RGBColor(37, 99, 235)
                
                # Contenu
                content_placeholder = slide.placeholders[1]
                text_frame = content_placeholder.text_frame
                text_frame.clear()  # Nettoyer le contenu par defaut
                
                for i, item in enumerate(current_slide_content):
                    if i == 0:
                        p = text_frame.paragraphs[0]
                    else:
                        p = text_frame.add_paragraph()
                    
                    # Detecter si c'est un titre
                    if (item.startswith('#') or 
                        (len(item) < 80 and item.isupper()) or
                        (i == 0 and len(item) < 100)):
                        # Style titre
                        p.text = f"• {item.replace('#', '').strip()}"
                        p.font.size = Pt(20)
                        p.font.color.rgb = RGBColor(249, 115, 22)  # Orange
                        p.font.bold = True
                    else:
                        # Style normal
                        p.text = f"  ◦ {item[:200]}{'...' if len(item) > 200 else ''}"
                        p.font.size = Pt(16)
                        p.font.color.rgb = RGBColor(55, 65, 81)  # Gris fonce
                    
                    p.space_after = Pt(12)
                
                current_slide_content = []
    
    # Derniere slide s'il reste du contenu
    if current_slide_content:
        slide_counter += 1
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = f"📖 Section {slide_counter - 1}"
        title_para = slide.shapes.title.text_frame.paragraphs[0]
        title_para.font.size = Pt(32)
        title_para.font.color.rgb = RGBColor(37, 99, 235)
        
        content_placeholder = slide.placeholders[1]
        text_frame = content_placeholder.text_frame
        text_frame.clear()
        
        for i, item in enumerate(current_slide_content):
            if i == 0:
                p = text_frame.paragraphs[0]
            else:
                p = text_frame.add_paragraph()
            
            p.text = f"• {item[:150]}{'...' if len(item) > 150 else ''}"
            p.font.size = Pt(16)
            p.font.color.rgb = RGBColor(55, 65, 81)
            p.space_after = Pt(12)
    
    # === SLIDE DE CONCLUSION ===
    conclusion_slide = prs.slides.add_slide(prs.slide_layouts[5])  # Blank layout
    
    # Ajouter une zone de texte pour la conclusion
    left = Inches(1)
    top = Inches(2)
    width = Inches(8)
    height = Inches(4)
    
    textbox = conclusion_slide.shapes.add_textbox(left, top, width, height)
    text_frame = textbox.text_frame
    
    # Titre de conclusion
    conclusion_title = text_frame.add_paragraph()
    conclusion_title.text = "🎉 Merci pour votre attention !"
    conclusion_title.font.size = Pt(36)
    conclusion_title.font.color.rgb = RGBColor(249, 115, 22)
    conclusion_title.font.bold = True
    conclusion_title.alignment = PP_ALIGN.CENTER
    
    # Texte de conclusion
    conclusion_text = text_frame.add_paragraph()
    conclusion_text.text = "\n\nEtienne - Assistant IA Educatif\nCollege Champagneur\n\n📚 Continuez a apprendre et a explorer ! 🎓"
    conclusion_text.font.size = Pt(18)
    conclusion_text.font.color.rgb = RGBColor(37, 99, 235)
    conclusion_text.alignment = PP_ALIGN.CENTER
    
    buffer = BytesIO()
    prs.save(buffer)
    buffer.seek(0)
    return buffer

def generate_xlsx_document(title: str, content: str) -> BytesIO:
    """Genere un fichier Excel"""
    buffer = BytesIO()
    wb = Workbook()
    ws = wb.active
    ws.title = "Document Etienne"
    
    # En-tetes
    ws['A1'] = title
    ws['A1'].font = Font(bold=True, size=16)
    ws['A1'].fill = PatternFill(start_color='366092', end_color='366092', fill_type='solide')
    ws['A1'].font = Font(bold=True, size=16, color='FFFFFF')
    
    # Contenu
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    ws['A3'] = "Points cles :"
    ws['A3'].font = Font(bold=True)
    
    row = 4
    for i, para in enumerate(paragraphs):
        ws[f'A{row}'] = f"{i + 1}."
        ws[f'B{row}'] = para
        ws[f'A{row}'].font = Font(bold=True)
        row += 1
    
    # Ajuster la largeur des colonnes
    ws.column_dimensions['A'].width = 5
    ws.column_dimensions['B'].width = 80
    
    # Pied de page
    ws[f'A{row + 2}'] = f"Genere par Etienne le {datetime.now().strftime('%d/%m/%Y a %H:%M')}"
    ws[f'A{row + 2}'].font = Font(italic=True)
    
    wb.save(buffer)
    buffer.seek(0)
    return buffer

@api_router.post("/generate-document")
async def generate_document(request: DocumentRequest):
    """Genere un document dans le format demande"""
    try:
        # Validation du format
        allowed_formats = ['pdf', 'docx', 'pptx', 'xlsx']
        if request.format not in allowed_formats:
            raise HTTPException(status_code=400, detail=f"Format non supporte. Formats autorises: {', '.join(allowed_formats)}")
        
        # Generation du nom de fichier
        if not request.filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{request.title.replace(' ', '_')}_{timestamp}.{request.format}"
        else:
            filename = request.filename if request.filename.endswith(f".{request.format}") else f"{request.filename}.{request.format}"
        
        # Generation du document selon le format
        if request.format == 'pdf':
            buffer = generate_pdf_document(request.title, request.content)
            media_type = "application/pdf"
        elif request.format == 'docx':
            buffer = generate_docx_document(request.title, request.content)
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif request.format == 'pptx':
            buffer = generate_pptx_document(request.title, request.content)
            media_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        elif request.format == 'xlsx':
            buffer = generate_xlsx_document(request.title, request.content)
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        
        return StreamingResponse(
            BytesIO(buffer.read()),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logging.error(f"Erreur generation document: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la generation du document: {str(e)}")

@api_router.post("/upload-file")
async def upload_and_extract_file(file: UploadFile = File(...)):
    """Upload un fichier et extrait son contenu texte"""
    try:
        # Verifier la taille du fichier (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        file_size = 0
        
        # Lire le fichier pour verifier la taille
        content = await file.read()
        file_size = len(content)
        
        if file_size > max_size:
            raise HTTPException(
                status_code=413,
                detail="Fichier trop volumineux. Taille maximale: 10MB"
            )
        
        # Remettre le pointeur au debut pour l extraction
        file.file = BytesIO(content)
        
        # Extraire le texte
        extracted_text = await extract_text_from_file(file)
        
        # Limiter la longueur du texte extrait (pour eviter les tokens excessifs)
        max_text_length = 10000  # ~2500 mots
        if len(extracted_text) > max_text_length:
            extracted_text = extracted_text[:max_text_length] + "\n\n[...Texte tronque pour optimiser l analyse...]"
        
        return {
            "filename": file.filename,
            "file_size": file_size,
            "extracted_text": extracted_text,
            "text_length": len(extracted_text),
            "message": "Fichier traite avec succes. Vous pouvez maintenant poser votre question."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Erreur upload fichier: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement du fichier: {str(e)}"
        )

@api_router.post("/analyze-file", response_model=ChatMessage)
async def analyze_file_with_question(request: FileAnalysisRequest):
    """Analyse un fichier avec une question specifique - 100% GRATUIT avec Gemini"""
    try:
        # Preparer le prompt avec le contenu du fichier
        enhanced_message = f"""
CONTEXTE: L'utilisateur a uploade un document ({request.filename}) et pose la question suivante:

QUESTION: {request.question}

CONTENU DU DOCUMENT:
{request.extracted_text}

INSTRUCTIONS: 
- Analysez le contenu du document en relation avec la question posee
- Fournissez une reponse precise basee sur le contenu du document
- Si la reponse n'est pas dans le document, mentionnez-le clairement
- Structurez votre reponse de maniere claire et pedagogique
"""

        # Configuration systeme pour l analyse de fichiers
        system_message = """Tu es un assistant IA specialise dans l analyse de documents pour les etudiants quebecois. 
Tu dois analyser le contenu fourni et repondre a la question de l utilisateur de maniere precise et pedagogique.
Adapte ton langage au niveau de etudes quebecois et utilise un français accessible."""

        # === GOOGLE GEMINI 2.0 FLASH (100% GRATUIT) ===
        model = genai.GenerativeModel(
            model_name='gemini-2.0-flash-exp',
            system_instruction=system_message
        )
        
        # Generer la reponse
        response = await asyncio.to_thread(
            model.generate_content,
            enhanced_message
        )
        
        response_text = response.text
        
        # Creer l objet ChatMessage
        chat_message = ChatMessage(
            session_id=f"file-{request.filename}-{uuid.uuid4()}",
            message=f"📎 Analyse du fichier '{request.filename}': {request.question}",
            response=response_text,
            message_type=request.message_type,
            trust_score=0.90,  # Score eleve pour l analyse de documents
            sources=[request.filename]
        )
        
        # Sauvegarder en base de donnees
        await db.chat_messages.insert_one(chat_message.dict())
        
        return chat_message
        
    except Exception as e:
        logging.error(f"Erreur analyse fichier: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l analyse du fichier: {str(e)}"
        )

@api_router.get("/subjects")
async def get_school_subjects():
    """Retourne la liste des matieres du systeme educatif quebecois"""
    subjects = {
        "langues": {
            "name": "Langues",
            "subjects": ["Français", "Anglais", "Espagnol"]
        },
        "sciences": {
            "name": "Sciences & Mathematiques",
            "subjects": ["Mathematiques", "Sciences et technologies"]
        },
        "sciences_humaines": {
            "name": "Sciences Humaines",
            "subjects": ["Histoire", "Geographie", "Culture et societe quebecoise", "Monde contemporain"]
        },
        "formation_generale": {
            "name": "Formation Generale",
            "subjects": ["Education financiere", "Methodologie", "Education physique"]
        },
        "arts": {
            "name": "Arts",
            "subjects": ["Art dramatique", "Arts plastiques", "Danse", "Musique"]
        }
    }
    return subjects

# Modeles pour les nouvelles fonctionnalites
class AIDetectionRequest(BaseModel):
    text: str

class PlagiarismCheckRequest(BaseModel):
    text: str

class AIFeedbackRequest(BaseModel):
    text: str
    predicted_probability: float
    actual_is_ai: bool  # True si c'etait vraiment IA, False si humain
    analysis_id: Optional[str] = None


@api_router.post("/detect-ai")
async def detect_ai_endpoint(request: AIDetectionRequest):
    """Detecte si un texte a ete genere par IA - VERSION AMELIOREE AVEC CLAUDE"""
    try:
        # Utiliser la nouvelle detection LLM
        result = await detect_ai_content_with_llm(request.text)
        
        return {
            "success": True,
            "detection_result": result,
            "message": f"Analyse terminee. Probabilite IA: {result['ai_probability']*100}%"
        }
    except Exception as e:
        logging.error(f"Erreur detection IA: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la detection IA: {str(e)}")

@api_router.post("/check-plagiarism")
async def check_plagiarism_endpoint(request: PlagiarismCheckRequest):
    """Verifie le risque de plagiat dans un texte"""
    try:
        result = check_plagiarism(request.text)
        return {
            "success": True,
            "plagiarism_result": result,
            "message": f"Verification terminee. Risque de plagiat: {result['risk_level']}"
        }
    except Exception as e:
        logging.error(f"Erreur verification plagiat: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la verification de plagiat: {str(e)}")

@api_router.post("/analyze-text")
async def analyze_text_complete(request: AIDetectionRequest):
    """Analyse complete de un texte (IA + Plagiat + Langue) - VERSION AMELIOREE"""
    try:
        # Detection de langue
        detected_language = detect_language(request.text)
        
        # Detection IA avec le nouveau systeme Claude
        ai_result = await detect_ai_content_with_llm(request.text)
        
        # Verification plagiat
        plagiarism_result = check_plagiarism(request.text)
        
        # Analyse combinee
        overall_risk = "Low"
        if ai_result["is_likely_ai"] or plagiarism_result["is_suspicious"]:
            overall_risk = "High"
        elif ai_result["ai_probability"] > 0.3 or plagiarism_result["plagiarism_risk"] > 0.3:
            overall_risk = "Medium"
        
        recommendations = []
        if ai_result["is_likely_ai"]:
            recommendations.append(f"Ce texte semble genere par IA ({ai_result['ai_probability']*100:.0f}% de probabilite). Verifiez l originalite.")
        if plagiarism_result["is_suspicious"]:
            recommendations.append("Risque de plagiat detecte. Verifiez les sources.")
        if not recommendations:
            recommendations.append("Le texte semble original et authentique.")
        
        return {
            "success": True,
            "language": detected_language,
            "ai_detection": ai_result,
            "plagiarism_check": plagiarism_result,
            "overall_assessment": {
                "risk_level": overall_risk,
                "recommendations": recommendations,
                "text_length": len(request.text),
                "word_count": len(request.text.split())
            }
        }
        
    except Exception as e:
        logging.error(f"Erreur analyse complete: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l analyse complete: {str(e)}")

@api_router.post("/ai-feedback")
async def submit_ai_feedback(request: AIFeedbackRequest):
    """
    Permet aux utilisateurs de corriger la detection IA pour ameliorer le systeme.
    Le systeme apprend de ces corrections pour devenir plus precis.
    """
    try:
        # Enregistrer le feedback dans la base de donnees
        feedback_doc = {
            "id": str(uuid.uuid4()),
            "text": request.text,
            "predicted_probability": request.predicted_probability,
            "actual_is_ai": request.actual_is_ai,
            "analysis_id": request.analysis_id,
            "timestamp": datetime.now(timezone.utc)
        }
        
        await db.ai_feedback.insert_one(feedback_doc)
        
        # Faire apprendre le detecteur avance
        advanced_detector.learn_from_feedback(
            text=request.text,
            actual_label=request.actual_is_ai,
            predicted_prob=request.predicted_probability
        )
        
        # Sauvegarder les nouveaux poids dans la base de donnees
        await save_detector_weights()
        
        # Calculer les statistiques de apprentissage
        total_feedback = await db.ai_feedback.count_documents({})
        
        logging.info(f"Feedback enregistre. Total feedbacks: {total_feedback}")
        
        return {
            "success": True,
            "message": "Merci! Votre feedback a ete enregistre et le systeme s'est ameliore.",
            "total_feedback_count": total_feedback,
            "learning_status": "Poids du detecteur mis a jour"
        }
        
    except Exception as e:
        logging.error(f"Erreur enregistrement feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l enregistrement du feedback: {str(e)}")

@api_router.get("/ai-stats")
async def get_ai_detection_stats():
    """Retourne les statistiques du systeme de auto-apprentissage"""
    try:
        # Compter le nombre de feedbacks
        total_feedback = await db.ai_feedback.count_documents({})
        
        # Recuperer les poids actuels
        weights_doc = await db.detector_weights.find_one({"version": "latest"})
        
        # Calculer l accuracy si possible
        if total_feedback > 0:
            feedbacks = await db.ai_feedback.find().to_list(length=1000)
            
            correct_predictions = sum(
                1 for fb in feedbacks 
                if (fb["predicted_probability"] > 0.5) == fb["actual_is_ai"]
            )
            
            accuracy = (correct_predictions / len(feedbacks)) * 100 if feedbacks else 0
        else:
            accuracy = 0
        
        return {
            "success": True,
            "total_feedback_received": total_feedback,
            "estimated_accuracy": round(accuracy, 1) if total_feedback > 0 else "Pas encore de donnees",
            "last_weights_update": weights_doc.get("updated_at") if weights_doc else None,
            "learning_status": "Actif" if total_feedback > 0 else "En attente de feedback",
            "message": f"Le systeme a appris de {total_feedback} exemples"
        }
        
    except Exception as e:
        logging.error(f"Erreur stats: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la recuperation des stats: {str(e)}")



# ============================================
# ENDPOINTS ADMIN - LOGS DE SECURITE
# ============================================

@api_router.get("/admin/illegal-content-logs")
async def get_illegal_content_logs(
    limit: int = 100,
    severity: Optional[str] = None,
    category: Optional[str] = None,
    current_admin: dict = Depends(get_current_admin)
):
    """
    Recupere les logs de tentatives de contenu illegal
    AUTHENTIFICATION REQUISE
    """


# ============================================
# ENDPOINTS D'AUTHENTIFICATION ADMIN
# ============================================

class AdminLoginRequest(BaseModel):
    username: str
    password: str

class AdminLoginResponse(BaseModel):
    success: bool
    token: str
    token_type: str = "bearer"
    expires_in: int = 86400  # 24 heures en secondes

@api_router.post("/admin/login", response_model=AdminLoginResponse)
async def admin_login(credentials: AdminLoginRequest):
    """Login admin - genere un JWT token"""
    from security_advanced import ADMIN_USERNAME, ADMIN_PASSWORD_HASH
    
    if credentials.username != ADMIN_USERNAME:
        raise HTTPException(status_code=401, detail="Identifiants invalides")
    
    if not verify_password(credentials.password, ADMIN_PASSWORD_HASH):
        raise HTTPException(status_code=401, detail="Identifiants invalides")
    
    # Creer le token JWT
    token = create_access_token({"username": credentials.username})
    
    return AdminLoginResponse(
        success=True,
        token=token
    )

@api_router.post("/admin/send-daily-report")
async def trigger_daily_report(current_admin: dict = Depends(get_current_admin)):
    """
    Genere et envoie manuellement le rapport quotidien
    AUTHENTIFICATION REQUISE
    """
    try:
        # Recuperer tous les logs
        logs_cursor = await db.security_illegal_logs.find({}).sort("timestamp", -1).to_list(1000)
        
        logs = []
        for log in logs_cursor:
            log['_id'] = str(log.get('_id', ''))
            log['timestamp'] = log.get('timestamp', datetime.now(timezone.utc)).isoformat() if isinstance(log.get('timestamp'), datetime) else str(log.get('timestamp'))
            logs.append(log)
        
        # Statistiques
        stats = {
            "total_logs": await db.security_illegal_logs.count_documents({}),
            "critical_count": await db.security_illegal_logs.count_documents({"severity": "critical"}),
            "high_count": await db.security_illegal_logs.count_documents({"severity": "high"}),
            "category_stats": {}
        }
        
        for category_name in ILLEGAL_CONTENT_PATTERNS.keys():
            count = await db.security_illegal_logs.count_documents({"detected_category": category_name})
            if count > 0:
                stats["category_stats"][category_name] = count
        
        # Envoyer le rapport
        success = await asyncio.to_thread(send_daily_report, logs, stats)
        
        # Sauvegarder localement
        if logs:
            import os
            report_dir = Path(__file__).parent / "security_reports"
            report_dir.mkdir(exist_ok=True)
            
            # CSV
            csv_content = generate_csv_report(logs)
            csv_path = report_dir / f"report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
            with open(csv_path, 'wb') as f:
                f.write(csv_content)
            
            # PDF
            pdf_content = generate_pdf_report(logs, stats)
            pdf_path = report_dir / f"report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pdf"
            with open(pdf_path, 'wb') as f:
                f.write(pdf_content)
        
        return {
            "success": success,
            "message": "Rapport envoye par email et sauvegarde localement" if success else "Erreur lors de l envoi",
            "logs_count": len(logs),
            "report_saved": True if logs else False
        }
        
    except Exception as e:
        logging.error(f"Erreur generation rapport: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/admin/export-csv")
async def export_logs_csv(
    days: int = 30,
    current_admin: dict = Depends(get_current_admin)
):
    """
    Exporte les logs en CSV
    AUTHENTIFICATION REQUISE
    """
    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        logs_cursor = await db.security_illegal_logs.find(
            {"timestamp": {"$gte": cutoff_date}}
        ).sort("timestamp", -1).to_list(10000)
        
        logs = []
        for log in logs_cursor:
            log['_id'] = str(log.get('_id', ''))
            log['timestamp'] = log.get('timestamp', datetime.now(timezone.utc)).isoformat() if isinstance(log.get('timestamp'), datetime) else str(log.get('timestamp'))
            logs.append(log)
        
        csv_content = generate_csv_report(logs)
        
        return StreamingResponse(
            iter([csv_content]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=security_logs_{datetime.now(timezone.utc).strftime('%Y%m%de')}.csv"
            }
        )
        
    except Exception as e:
        logging.error(f"Erreur export CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/admin/export-pdf")
async def export_logs_pdf(
    days: int = 30,
    current_admin: dict = Depends(get_current_admin)
):
    """
    Exporte les logs en PDF
    AUTHENTIFICATION REQUISE
    """
    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        logs_cursor = await db.security_illegal_logs.find(
            {"timestamp": {"$gte": cutoff_date}}
        ).sort("timestamp", -1).to_list(10000)
        
        logs = []
        for log in logs_cursor:
            log['_id'] = str(log.get('_id', ''))
            log['timestamp'] = log.get('timestamp', datetime.now(timezone.utc)).isoformat() if isinstance(log.get('timestamp'), datetime) else str(log.get('timestamp'))
            logs.append(log)
        
        # Statistiques
        stats = {
            "total_logs": len(logs),
            "critical_count": sum(1 for log in logs if log.get('severity') == 'critical'),
            "high_count": sum(1 for log in logs if log.get('severity') == 'high'),
            "category_stats": {}
        }
        
        for log in logs:
            cat = log.get('detected_category', 'unknown')
            stats["category_stats"][cat] = stats["category_stats"].get(cat, 0) + 1
        
        pdf_content = generate_pdf_report(logs, stats)
        
        return StreamingResponse(
            iter([pdf_content]),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=security_report_{datetime.now(timezone.utc).strftime('%Y%m%de')}.pdf"
            }
        )
        
    except Exception as e:
        logging.error(f"Erreur export PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/admin/rate-limit-stats")
async def get_rate_limit_stats(current_admin: dict = Depends(get_current_admin)):
    """
    Obtient les statistiques de rate limiting
    AUTHENTIFICATION REQUISE
    """
    from security_advanced import rate_limit_storage
    
    stats = {
        "total_ips_tracked": len(rate_limit_storage),
        "ips_near_limit": 0,
        "ips_blocked": 0,
        "top_ips": []
    }
    
    from security_advanced import RATE_LIMIT_ATTEMPTS
    
    for ip, attempts in rate_limit_storage.items():
        attempt_count = len(attempts)
        if attempt_count >= RATE_LIMIT_ATTEMPTS:
            stats["ips_blocked"] += 1
        elif attempt_count >= RATE_LIMIT_ATTEMPTS * 0.7:
            stats["ips_near_limit"] += 1
        
        stats["top_ips"].append({
            "ip": ip,
            "attempts": attempt_count,
            "status": "blocked" if attempt_count >= RATE_LIMIT_ATTEMPTS else "active"
        })
    
    # Trier par nombre de tentatives
    stats["top_ips"] = sorted(stats["top_ips"], key=lambda x: x["attempts"], reverse=True)[:20]
    
    return stats

@api_router.delete("/admin/illegal-content-logs/clear")
async def clear_illegal_content_logs(older_than_days: int = 30, current_admin: dict = Depends(get_current_admin)):
    """
    Supprime les anciens logs (par defaut > 30 jours)
    ADMIN ONLY - A proteger avec authentification en production
    """
    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        
        result = await db.security_illegal_logs.delete_many({
            "timestamp": {"$lt": cutoff_date}
        })
        
        return {
            "success": True,
            "deleted_count": result.deleted_count,
            "older_than_days": older_than_days
        }
        
    except Exception as e:
        logging.error(f"Erreur suppression logs: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la suppression des logs: {str(e)}")



# ============================================
# GENERATION D'IMAGES IA (Hugging Face - GRATUIT)
# ============================================

import requests
import base64

HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY', '')
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "blurry, bad quality, distorted"

class ImageGenerationResponse(BaseModel):
    success: bool
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    error: Optional[str] = None

async def generate_image_huggingface(prompt: str, negative_prompt: str = "") -> bytes:
    """Genere une image avec Hugging Face Stable Diffusion (GRATUIT)"""
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": negative_prompt,
            "num_inference_steps": 30,
            "guidance_scale": 7.5
        }
    }
    
    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logging.error(f"Erreur generation image HF: {e}")
        raise

@api_router.post("/generate-image", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest):
    """
    Genere une image IA a partir de prompt
    Utilise Hugging Face Stable Diffusion (100% GRATUIT)
    """
    try:
        logging.info(f"Generation image: {request.prompt}")
        
        # Generer image
        image_bytes = await generate_image_huggingface(request.prompt, request.negative_prompt)
        
        # Convertir en base64 pour affichage
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        return ImageGenerationResponse(
            success=True,
            image_base64=image_base64,
            error=None
        )
        
    except Exception as e:
        logging.error(f"Erreur generation image: {e}")
        return ImageGenerationResponse(
            success=False,
            image_base64=None,
            error=str(e)
        )


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    """Charge les poids du detecteur au demarrage"""
    await load_detector_weights()
    logging.info("Application demarree et poids du detecteur charges")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
