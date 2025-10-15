```python

"""

Étienne API - Assistant IA Éducatif

Backend FastAPI pour Render.com

Version 2.0

"""

from fastapi import FastAPI, HTTPException

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from typing import Optional

import uuid

import os

app = FastAPI(title="Étienne API", version="2.0")

app.add_middleware(

CORSMiddleware,

allow_origins=["*"],

allow_credentials=True,

allow_methods=["*"],

allow_headers=["*"],

)

class ChatRequest(BaseModel):

message: str

message_type: str

session_id: Optional[str] = None

class TextAnalysisRequest(BaseModel):

text: str

analysis_type: str = "complete"

ENGLISH_SOURCES = {

"grammar": [

"Oxford English Grammar (oxford.com)",

"Cambridge Grammar (cambridge.org)",

"BBC Learning English (bbc.co.uk/learningenglish)"

],

"literature": [

"CliffsNotes Literature Guides (cliffsnotes.com)",

"SparkNotes Literature (sparknotes.com)",

"Lecturia Academic Library (lecturia.com)"

],

"academic": [

"Purdue OWL Writing Lab (owl.purdue.edu)",

"Harvard Writing Center (writingcenter.fas.harvard.edu)",

"CliffsNotes Study Guides (cliffsnotes.com/study-guides)"

]

}

QUEBEC_SOURCES = [

"Gouvernement du Québec (quebec.ca)",

"MEES - Ministère de l'Éducation (education.gouv.qc.ca)",

"Allô Prof (alloprof.qc.ca)",

"BAnQ - Bibliothèque nationale (banq.qc.ca)"

]

def detect_language(text: str) -> str:

english_words = ["the", "and", "to", "of", "help", "english", "grammar", "literature"]

french_words = ["le", "de", "et", "à", "aide", "français", "grammaire", "littérature"]

text_lower = text.lower()

eng_count = sum(1 for word in english_words if word in text_lower)

fr_count = sum(1 for word in french_words if word in text_lower)

return "en" if eng_count > fr_count and eng_count > 0 else "fr"

def detect_ai(text: str) -> dict:

ai_patterns = [

"as an ai", "i'm an ai", "as a language model",

"i don't have personal", "i cannot", "however",

"furthermore", "moreover"

]

text_lower = text.lower()

detected = [p for p in ai_patterns if p in text_lower]

score = len(detected) * 0.25

probability = min(round(score, 2), 0.99)

return {

"ai_probability": probability,

"is_likely_ai": probability > 0.5,

"confidence": "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low",

"detected_patterns": detected[:5]

}

def check_plagiarism(text: str) -> dict:

common_phrases = [

"according to the study", "research shows that",

"studies have shown", "experts agree that",

"the data suggests", "selon l'étude"

]

text_lower = text.lower()

found = [p for p in common_phrases if p in text_lower]

score = len(found) * 0.15

words = text_lower.split()

diversity = len(set(words)) / len(words) if words else 1.0

if diversity < 0.6:

score += 0.2

risk = min(round(score, 2), 0.99)

return {

"plagiarism_risk": risk,

"is_suspicious": risk > 0.4,

"vocabulary_diversity": round(diversity, 2),

"risk_level": "High" if risk > 0.6 else "Medium" if risk > 0.3 else "Low",

"suspicious_phrases": found,

"recommendation": "Vérifier l'originalité" if risk > 0.4 else "Contenu semble original"

}

def get_ai_response(message: str) -> dict:

detected_lang = detect_language(message)

message_lower = message.lower()

is_english_query = any(word in message_lower for word in [

"english", "anglais", "grammar", "literature", "writing"

])


if is_english_query:

if "literature" in message_lower:

category = "literature"

elif "writing" in message_lower or "essay" in message_lower:

category = "academic"

else:

category = "grammar"


sources = ENGLISH_SOURCES[category]


if detected_lang == "en":

response = f"For {category}, here are reliable academic sources:\n\n"

else:

response = f"Pour l'anglais ({category}), voici des sources académiques:\n\n"


for i, source in enumerate(sources, 1):

response += f"{i}. {source}\n"


response += "\n💡 Tip: Cross-reference multiple sources for best results."

trust_score = 0.95

else:

sources = QUEBEC_SOURCES


if detected_lang == "fr":

response = "Bonjour ! Je suis Étienne, votre assistant IA du Collège Champagneur.\n\n"

response += "Voici des ressources éducatives québécoises fiables:\n\n"

else:

response = "Hello! I'm Étienne, your AI assistant from Collège Champagneur.\n\n"

response += "Here are reliable Quebec educational resources:\n\n"


for i, source in enumerate(sources, 1):

response += f"{i}. {source}\n"


response += "\n💡 Conseil: Privilégiez les sources .gouv.qc.ca pour vos recherches."

trust_score = 0.85


return {

"response": response,

"trust_score": trust_score,

"sources": sources,

"detected_language": detected_lang

}

@app.get("/")

def root():

return {

"message": "Étienne API - Assistant IA Éducatif",

"status": "active",

"platform": "render"

}

@app.get("/api")

def api_root():

return {

"message": "Étienne API - Assistant IA Éducatif",

"status": "active",

"platform": "render",

"backend": "working",

"version": "2.0",

"college": "Collège Champagneur",

"features": [

"chat",

"sources_anglaises",

"detection_ia",

"verification_plagiat",

"analyse_complete"

],

"endpoints": {

"health": "GET /api/health",

"chat": "POST /api/chat",

"subjects": "GET /api/subjects",

"analyze": "POST /api/analyze-text",

"detect_ai": "POST /api/detect-ai",

"plagiarism": "POST /api/check-plagiarism"

}

}

@app.get("/api/health")

def health():

return {

"status": "healthy",

"platform": "render",

"version": "2.0",

"uptime": "operational"

}

@app.post("/api/chat")

def chat(request: ChatRequest):

try:

session_id = request.session_id or str(uuid.uuid4())

ai_result = get_ai_response(request.message)


return {

"id": str(uuid.uuid4()),

"session_id": session_id,

"message": request.message,

"response": ai_result["response"],

"message_type": request.message_type,

"trust_score": ai_result["trust_score"],

"sources": ai_result["sources"],

"timestamp": "2025-01-15T12:00:00Z"

}

except Exception as e:

raise HTTPException(status_code=500, detail=f"Erreur chat: {str(e)}")

@app.get("/api/subjects")

def subjects():

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

def analyze(request: TextAnalysisRequest):

try:

ai_result = detect_ai(request.text)

plagiarism_result = check_plagiarism(request.text)

lang = detect_language(request.text)


return {

"text_length": len(request.text),

"word_count": len(request.text.split()),

"detected_language": lang,

"ai_detection": ai_result,

"plagiarism_check": plagiarism_result,

"overall_assessment": {

"is_authentic": not ai_result["is_likely_ai"] and not plagiarism_result["is_suspicious"],

"confidence_score": round((1 - ai_result["ai_probability"] + (1 - plagiarism_result["plagiarism_risk"])) / 2, 2),

"recommendation": "Texte semble authentique" if not ai_result["is_likely_ai"] and not plagiarism_result["is_suspicious"] else "Vérification approfondie recommandée"

}

}

except Exception as e:

raise HTTPException(status_code=500, detail=f"Erreur analyse: {str(e)}")

@app.post("/api/detect-ai")

def detect_ai_endpoint(request: TextAnalysisRequest):

try:

result = detect_ai(request.text)

return result

except Exception as e:

raise HTTPException(status_code=500, detail=f"Erreur détection IA: {str(e)}")

@app.post("/api/check-plagiarism")

def plagiarism_endpoint(request: TextAnalysisRequest):

try:

result = check_plagiarism(request.text)

return result

except Exception as e:

raise HTTPException(status_code=500, detail=f"Erreur vérification plagiat: {str(e)}")

if __name__ == "__main__":

import uvicorn

port = int(os.environ.get("PORT", 10000))

uvicorn.run(app, host="0.0.0.0", port=port)

```
