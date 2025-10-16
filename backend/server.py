from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime, timezone
from emergentintegrations.llm.chat import LlmChat, UserMessage
import asyncio
import re
from fastapi.responses import StreamingResponse
from fastapi import UploadFile, File, Form
from io import BytesIO
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


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

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
    title: str = "Document Étienne"
    format: str = "pdf"  # pdf, docx, pptx, xlsx
    filename: Optional[str] = None

class TextAnalysisRequest(BaseModel):
    text: str
    analysis_type: str = "complete"  # "ai_detection", "plagiarism", "complete"

class FileAnalysisRequest(BaseModel):
    question: str
    extracted_text: str
    filename: str
    message_type: str = "je_veux"

# Système de confiance des sources
TRUSTED_DOMAINS = {
    ".gouv.qc.ca": 0.98,
    ".gouv.ca": 0.95,
    ".edu": 0.90,
    "quebec.ca": 0.97,
    "education.gouv.qc.ca": 0.98,
    "mees.gouv.qc.ca": 0.98,  # Ministère de l'Éducation du Québec
    "banq.qc.ca": 0.88,  # Bibliothèque nationale du Québec
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
    """Calcule le score de confiance d'une source basé sur le domaine et le contenu"""
    base_score = 0.5
    
    # Vérification du domaine
    for domain, score in TRUSTED_DOMAINS.items():
        if domain in url.lower():
            base_score = max(base_score, score)
    
    # Analyse basique du contenu si fourni
    if content:
        quality_indicators = [
            "bibliographie", "références", "source", "étude", "recherche", 
            "académique", "officiel", "ministère", "université", "peer-review"
        ]
        quality_count = sum(1 for indicator in quality_indicators if indicator in content.lower())
        content_bonus = min(0.15, quality_count * 0.03)
        base_score = min(0.98, base_score + content_bonus)
    
    return round(base_score, 2)

async def extract_text_from_file(file: UploadFile) -> str:
    """Extrait le texte d'un fichier uploadé"""
    try:
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        extracted_text = ""
        file_extension = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
        
        try:
            if file_extension == 'pdf':
                # Extraction PDF avec pdfplumber (meilleure qualité)
                with pdfplumber.open(temp_file_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            extracted_text += text + "\n"
                
                # Fallback avec PyPDF2 si pdfplumber échoue
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
                    df = pd.read_excel(temp_file_path, engine='xlrd')
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
                    detail=f"Format de fichier non supporté: {file_extension}. "
                          f"Formats acceptés: PDF, DOCX, TXT, XLSX, CSV, PPTX"
                )
        
        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
        if not extracted_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Impossible d'extraire le texte de ce fichier. Vérifiez que le fichier n'est pas protégé ou corrompu."
            )
        
        return extracted_text.strip()
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Erreur extraction texte: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'extraction du texte: {str(e)}"
        )

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

async def get_ai_response(message: str, message_type: str) -> dict:
    """Obtient une réponse d'Étienne selon le type de message"""
    try:
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
        
        # Configuration du système selon le type - Étienne
        system_messages = {
            "je_veux": "Tu es Étienne, un assistant pédagogique spécialisé pour les étudiants québécois. Réponds de manière claire et éducative. Pour l'anglais, recommande les meilleures sources mondiales comme Oxford, Cambridge, BBC Learning, Purdue OWL. Utilise un français québécois accessible.",
            "je_recherche": "Tu es Étienne, assistant de recherche éducative. Aide les étudiants québécois à explorer des sujets scolaires. Pour l'anglais, oriente vers des sources internationales prestigieuses. Propose des pistes de recherche pédagogiques.",
            "sources_fiables": "Tu es Étienne, expert en sources académiques. Guide vers des sources fiables: pour le québécois (.gouv.qc.ca, .edu), pour l'anglais (Oxford, Cambridge, BBC, Purdue OWL, Harvard, MIT). Explique comment évaluer la crédibilité.",
            "activites": "Tu es Étienne, créateur d'activités pédagogiques pour étudiants québécois. Propose des exercices engageants adaptés au programme québécois. Pour l'anglais, utilise des ressources internationales de qualité."
        }
        
        # Détection de questions sur l'anglais
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
                english_category = "grammar"  # Par défaut
            
            sources_to_add = english_sources[english_category][:3]  # Top 3 sources
        
        # Détection de la langue du message
        detected_language = detect_language(message)
        
        # Enrichir le message si c'est une question d'anglais
        enhanced_message = message
        if is_english_query and english_category:
            enhanced_message = f"{message}\n\nNote: Je recommande particulièrement ces sources pour l'anglais: {', '.join(sources_to_add[:2])}."
        
        # Adapter le système selon la langue détectée
        base_system_message = system_messages.get(message_type, system_messages["je_veux"])
        
        if detected_language == "en":
            # Répondre en anglais si l'utilisateur écrit en anglais
            english_system_messages = {
                "je_veux": "You are Étienne, an educational assistant specialized for Quebec students. Respond clearly and educationally in English. For English topics, recommend the best global sources like Oxford, Cambridge, BBC Learning, Purdue OWL. Be helpful and accessible.",
                "je_recherche": "You are Étienne, an educational research assistant. Help Quebec students explore academic topics in English. Guide them to prestigious international sources. Suggest educational research paths.",
                "sources_fiables": "You are Étienne, an expert in academic sources. Guide to reliable sources: for Quebec (.gouv.qc.ca, .edu), for English (Oxford, Cambridge, BBC, Purdue OWL, Harvard, MIT). Explain how to evaluate credibility.",
                "activites": "You are Étienne, creator of educational activities for Quebec students. Propose engaging exercises in English adapted to Quebec curriculum. For English topics, use quality international resources."
            }
            system_message = english_system_messages.get(message_type, english_system_messages["je_veux"])
        else:
            system_message = base_system_message
        
        # Initialisation du chat Claude - Étienne
        chat = LlmChat(
            api_key=os.environ.get('EMERGENT_LLM_KEY'),
            session_id=f"etienne-{message_type}",
            system_message=system_message
        ).with_model("anthropic", "claude-4-sonnet-20250514")
        
        # Vérifier si l'utilisateur demande un document
        document_keywords = [
            "créer", "génère", "document", "fichier", "pdf", "word", "excel", "powerpoint",
            "télécharger", "exporter", "rapport", "résumé", "fiche", "présentation"
        ]
        
        wants_document = any(keyword in message.lower() for keyword in document_keywords)
        
        # Modifier le prompt si document demandé ou question anglais
        if wants_document:
            enhanced_message = f"{enhanced_message}\n\nNote: L'utilisateur semble vouloir créer un document. Structurez votre réponse de manière claire avec des titres et des points clés qui pourront être facilement exportés en PDF, Word, PowerPoint ou Excel."
        
        user_message = UserMessage(text=enhanced_message)
            
        response = await chat.send_message(user_message)
        
        # Ajouter sources spécialisées si anglais
        final_sources = sources_to_add if is_english_query else ["Sources éducatives québécoises recommandées"]
        trust_score = 0.95 if (message_type == "sources_fiables" or is_english_query) else None
        
        return {
            "response": response,
            "trust_score": trust_score,
            "sources": final_sources,
            "can_download": len(response) > 50
        }
        
    except Exception as e:
        logging.error(f"Erreur IA: {e}")
        return {
            "response": "Désolé, une erreur s'est produite. Veuillez réessayer.",
            "trust_score": None,
            "sources": []
        }

# Routes API
@api_router.get("/")
async def root():
    return {"message": "API Étienne - Assistant IA pour les étudiants québécois fourni par le Collège Champagneur"}

@api_router.post("/chat", response_model=ChatMessage)
async def chat_with_ai(request: ChatRequest):
    """Endpoint principal pour le chat avec l'IA"""
    try:
        # Génération d'un session_id si non fourni
        session_id = request.session_id or str(uuid.uuid4())
        
        # Obtention de la réponse IA
        ai_result = await get_ai_response(request.message, request.message_type)
        
        # Création de l'objet ChatMessage
        chat_message = ChatMessage(
            session_id=session_id,
            message=request.message,
            response=ai_result["response"],
            message_type=request.message_type,
            trust_score=ai_result["trust_score"],
            sources=ai_result["sources"]
        )
        
        # Sauvegarde en base de données
        await db.chat_messages.insert_one(chat_message.dict())
        
        return chat_message
        
    except Exception as e:
        logging.error(f"Erreur chat: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors du traitement de la demande")

@api_router.get("/chat/history/{session_id}", response_model=List[ChatMessage])
async def get_chat_history(session_id: str):
    """Récupère l'historique d'une session de chat"""
    try:
        messages = await db.chat_messages.find(
            {"session_id": session_id}
        ).sort("timestamp", 1).to_list(100)
        
        return [ChatMessage(**message) for message in messages]
        
    except Exception as e:
        logging.error(f"Erreur historique: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération de l'historique")

@api_router.post("/sources/analyze")
async def analyze_sources(sources: List[str]):
    """Analyse la fiabilité d'une liste de sources"""
    try:
        analyzed_sources = []
        
        for source_url in sources:
            trust_score = calculate_trust_score(source_url)
            analyzed_sources.append({
                "url": source_url,
                "trust_score": trust_score,
                "trust_level": "Très fiable" if trust_score >= 0.8 else 
                              "Fiable" if trust_score >= 0.6 else 
                              "Modérément fiable" if trust_score >= 0.4 else "Peu fiable",
                "recommendation": "Source recommandée" if trust_score >= 0.7 else 
                                "Vérifier avec d'autres sources" if trust_score >= 0.5 else 
                                "Source non recommandée"
            })
        
        return {"analyzed_sources": analyzed_sources}
        
    except Exception as e:
        logging.error(f"Erreur analyse sources: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de l'analyse des sources")

def generate_pdf_document(title: str, content: str) -> BytesIO:
    """Génère un document PDF avec belle présentation"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    
    # Styles personnalisés
    from reportlab.lib.colors import HexColor, black, white
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.platypus import PageBreak
    
    # Style de titre personnalisé
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
    
    # En-tête avec logo Étienne
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
    
    # Ligne de séparation
    from reportlab.platypus import HRFlowable
    story.append(HRFlowable(width="100%", thickness=2, color=HexColor('#f97316')))
    story.append(Spacer(1, 20))
    
    # Traitement du contenu
    sections = content.split('\n\n')
    for i, section in enumerate(sections):
        if section.strip():
            # Détecter si c'est un titre (commence par #, ou est court et en majuscules)
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
    
    # Footer élégant
    story.append(Spacer(1, 30))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#e5e7eb')))
    story.append(Spacer(1, 10))
    
    footer_table = Table([
        ['Généré par Étienne', f"{datetime.now().strftime('%d/%m/%Y à %H:%M')}", 'Collège Champagneur'],
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
    """Génère un document Word avec belle présentation"""
    doc = Document()
    
    # En-tête avec logo Étienne
    header_para = doc.add_paragraph()
    header_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    header_para.add_run("🎓 ")
    title_run = header_para.add_run(title)
    title_run.bold = True
    title_run.font.size = DocxPt(20)
    title_run.font.color.rgb = RGBColor(249, 115, 22)  # Orange
    header_para.add_run(" 📚")
    
    # Ligne de séparation
    separator_para = doc.add_paragraph()
    separator_run = separator_para.add_run("_" * 80)
    separator_run.font.color.rgb = RGBColor(249, 115, 22)
    
    doc.add_paragraph()  # Espace
    
    # Traitement du contenu
    sections = content.split('\n\n')
    for i, section in enumerate(sections):
        if section.strip():
            # Détecter si c'est un titre
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
    
    # Footer élégant
    doc.add_paragraph()
    footer_separator = doc.add_paragraph()
    footer_separator_run = footer_separator.add_run("─" * 80)
    footer_separator_run.font.color.rgb = RGBColor(229, 231, 235)  # Gris clair
    
    footer_para = doc.add_paragraph()
    footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    footer_left = footer_para.add_run("Généré par Étienne")
    footer_left.font.size = DocxPt(9)
    footer_left.font.color.rgb = RGBColor(107, 114, 128)
    
    footer_center = footer_para.add_run(f" • {datetime.now().strftime('%d/%m/%Y à %H:%M')} • ")
    footer_center.font.size = DocxPt(9)
    footer_center.font.color.rgb = RGBColor(107, 114, 128)
    
    footer_right = footer_para.add_run("Collège Champagneur")
    footer_right.font.size = DocxPt(9)
    footer_right.font.color.rgb = RGBColor(107, 114, 128)
    footer_right.italic = True
    
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

def generate_pptx_document(title: str, content: str) -> BytesIO:
    """Génère une belle présentation PowerPoint"""
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
    subtitle_shape.text = f"Présenté par Étienne\nAssistant IA Éducatif\nCollège Champagneur\n\n{datetime.now().strftime('%d %B %Y')}"
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
            
            # Créer une nouvelle slide tous les 4 points ou si section trop longue
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
                text_frame.clear()  # Nettoyer le contenu par défaut
                
                for i, item in enumerate(current_slide_content):
                    if i == 0:
                        p = text_frame.paragraphs[0]
                    else:
                        p = text_frame.add_paragraph()
                    
                    # Détecter si c'est un titre
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
                        p.font.color.rgb = RGBColor(55, 65, 81)  # Gris foncé
                    
                    p.space_after = Pt(12)
                
                current_slide_content = []
    
    # Dernière slide s'il reste du contenu
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
    conclusion_text.text = "\n\nÉtienne - Assistant IA Éducatif\nCollège Champagneur\n\n📚 Continuez à apprendre et à explorer ! 🎓"
    conclusion_text.font.size = Pt(18)
    conclusion_text.font.color.rgb = RGBColor(37, 99, 235)
    conclusion_text.alignment = PP_ALIGN.CENTER
    
    buffer = BytesIO()
    prs.save(buffer)
    buffer.seek(0)
    return buffer

def generate_xlsx_document(title: str, content: str) -> BytesIO:
    """Génère un fichier Excel"""
    buffer = BytesIO()
    wb = Workbook()
    ws = wb.active
    ws.title = "Document Étienne"
    
    # En-têtes
    ws['A1'] = title
    ws['A1'].font = Font(bold=True, size=16)
    ws['A1'].fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
    ws['A1'].font = Font(bold=True, size=16, color='FFFFFF')
    
    # Contenu
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    ws['A3'] = "Points clés :"
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
    ws[f'A{row + 2}'] = f"Généré par Étienne le {datetime.now().strftime('%d/%m/%Y à %H:%M')}"
    ws[f'A{row + 2}'].font = Font(italic=True)
    
    wb.save(buffer)
    buffer.seek(0)
    return buffer

@api_router.post("/generate-document")
async def generate_document(request: DocumentRequest):
    """Génère un document dans le format demandé"""
    try:
        # Validation du format
        allowed_formats = ['pdf', 'docx', 'pptx', 'xlsx']
        if request.format not in allowed_formats:
            raise HTTPException(status_code=400, detail=f"Format non supporté. Formats autorisés: {', '.join(allowed_formats)}")
        
        # Génération du nom de fichier
        if not request.filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{request.title.replace(' ', '_')}_{timestamp}.{request.format}"
        else:
            filename = request.filename if request.filename.endswith(f".{request.format}") else f"{request.filename}.{request.format}"
        
        # Génération du document selon le format
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
        logging.error(f"Erreur génération document: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération du document: {str(e)}")

@api_router.post("/upload-file")
async def upload_and_extract_file(file: UploadFile = File(...)):
    """Upload un fichier et extrait son contenu texte"""
    try:
        # Vérifier la taille du fichier (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        file_size = 0
        
        # Lire le fichier pour vérifier la taille
        content = await file.read()
        file_size = len(content)
        
        if file_size > max_size:
            raise HTTPException(
                status_code=413,
                detail="Fichier trop volumineux. Taille maximale: 10MB"
            )
        
        # Remettre le pointeur au début pour l'extraction
        file.file = BytesIO(content)
        
        # Extraire le texte
        extracted_text = await extract_text_from_file(file)
        
        # Limiter la longueur du texte extrait (pour éviter les tokens excessifs)
        max_text_length = 10000  # ~2500 mots
        if len(extracted_text) > max_text_length:
            extracted_text = extracted_text[:max_text_length] + "\n\n[...Texte tronqué pour optimiser l'analyse...]"
        
        return {
            "filename": file.filename,
            "file_size": file_size,
            "extracted_text": extracted_text,
            "text_length": len(extracted_text),
            "message": "Fichier traité avec succès. Vous pouvez maintenant poser votre question."
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
    """Analyse un fichier avec une question spécifique"""
    try:
        # Préparer le prompt avec le contenu du fichier
        enhanced_message = f"""
CONTEXTE: L'utilisateur a uploadé un document ({request.filename}) et pose la question suivante:

QUESTION: {request.question}

CONTENU DU DOCUMENT:
{request.extracted_text}

INSTRUCTIONS: 
- Analysez le contenu du document en relation avec la question posée
- Fournissez une réponse précise basée sur le contenu du document
- Si la réponse n'est pas dans le document, mentionnez-le clairement
- Structurez votre réponse de manière claire et pédagogique
"""

        # Configuration système pour l'analyse de fichiers
        system_message = """Tu es un assistant IA spécialisé dans l'analyse de documents pour les étudiants québécois. 
Tu dois analyser le contenu fourni et répondre à la question de l'utilisateur de manière précise et pédagogique.
Adapte ton langage au niveau d'études québécois et utilise un français accessible."""

        # Initialisation du chat Claude avec Gemini pour les fichiers
        chat = LlmChat(
            api_key=os.environ.get('EMERGENT_LLM_KEY'),
            session_id=f"file-analysis-{uuid.uuid4()}",
            system_message=system_message
        ).with_model("gemini", "gemini-2.0-flash")  # Gemini est optimal pour l'analyse de documents
        
        # Envoi du message
        user_message = UserMessage(text=enhanced_message)
        response = await chat.send_message(user_message)
        
        # Créer l'objet ChatMessage
        chat_message = ChatMessage(
            session_id=f"file-{request.filename}-{uuid.uuid4()}",
            message=f"📎 Analyse du fichier '{request.filename}': {request.question}",
            response=response,
            message_type=request.message_type,
            trust_score=0.90,  # Score élevé pour l'analyse de documents
            sources=[request.filename]
        )
        
        # Sauvegarder en base de données
        await db.chat_messages.insert_one(chat_message.dict())
        
        return chat_message
        
    except Exception as e:
        logging.error(f"Erreur analyse fichier: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'analyse du fichier: {str(e)}"
        )

@api_router.get("/subjects")
async def get_school_subjects():
    """Retourne la liste des matières du système éducatif québécois"""
    subjects = {
        "langues": {
            "name": "Langues",
            "subjects": ["Français", "Anglais", "Espagnol"]
        },
        "sciences": {
            "name": "Sciences & Mathématiques",
            "subjects": ["Mathématiques", "Sciences et technologies"]
        },
        "sciences_humaines": {
            "name": "Sciences Humaines",
            "subjects": ["Histoire", "Géographie", "Culture et société québécoise", "Monde contemporain"]
        },
        "formation_generale": {
            "name": "Formation Générale",
            "subjects": ["Éducation financière", "Méthodologie", "Éducation physique"]
        },
        "arts": {
            "name": "Arts",
            "subjects": ["Art dramatique", "Arts plastiques", "Danse", "Musique"]
        }
    }
    return subjects

# Modèles pour les nouvelles fonctionnalités
class AIDetectionRequest(BaseModel):
    text: str

class PlagiarismCheckRequest(BaseModel):
    text: str

@api_router.post("/detect-ai")
async def detect_ai_endpoint(request: AIDetectionRequest):
    """Détecte si un texte a été généré par IA"""
    try:
        result = detect_ai_content(request.text)
        return {
            "success": True,
            "detection_result": result,
            "message": f"Analyse terminée. Probabilité IA: {result['ai_probability']*100}%"
        }
    except Exception as e:
        logging.error(f"Erreur détection IA: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la détection IA: {str(e)}")

@api_router.post("/check-plagiarism")
async def check_plagiarism_endpoint(request: PlagiarismCheckRequest):
    """Vérifie le risque de plagiat dans un texte"""
    try:
        result = check_plagiarism(request.text)
        return {
            "success": True,
            "plagiarism_result": result,
            "message": f"Vérification terminée. Risque de plagiat: {result['risk_level']}"
        }
    except Exception as e:
        logging.error(f"Erreur vérification plagiat: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la vérification de plagiat: {str(e)}")

@api_router.post("/analyze-text")
async def analyze_text_complete(request: AIDetectionRequest):
    """Analyse complète d'un texte (IA + Plagiat + Langue)"""
    try:
        # Détection de langue
        detected_language = detect_language(request.text)
        
        # Détection IA
        ai_result = detect_ai_content(request.text)
        
        # Vérification plagiat
        plagiarism_result = check_plagiarism(request.text)
        
        # Analyse combinée
        overall_risk = "Low"
        if ai_result["is_likely_ai"] or plagiarism_result["is_suspicious"]:
            overall_risk = "High"
        elif ai_result["ai_probability"] > 0.3 or plagiarism_result["plagiarism_risk"] > 0.3:
            overall_risk = "Medium"
        
        recommendations = []
        if ai_result["is_likely_ai"]:
            recommendations.append("Ce texte semble généré par IA. Vérifiez l'originalité.")
        if plagiarism_result["is_suspicious"]:
            recommendations.append("Risque de plagiat détecté. Vérifiez les sources.")
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
        logging.error(f"Erreur analyse complète: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse complète: {str(e)}")

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

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

