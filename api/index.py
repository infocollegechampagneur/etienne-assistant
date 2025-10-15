from http.server import BaseHTTPRequestHandler
import json
import os

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, Accept')
        self.send_header('Access-Control-Max-Age', '3600')
        self.end_headers()

    def do_GET(self):
        """Handle GET requests"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        try:
            # Route /api ou /api/
            if self.path in ['/api', '/api/']:
                response = {
                    "message": "Étienne API - Assistant IA Éducatif", 
                    "status": "active",
                    "platform": "vercel",
                    "backend": "working",
                    "version": "2.0",
                    "features": ["chat", "sources_anglaises", "detection_ia", "verification_plagiat"]
                }
            
            # Route /api/subjects
            elif self.path == '/api/subjects':
                response = {
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
                        "subjects": ["Histoire", "Géographie", "Monde contemporain"]
                    },
                    "arts": {
                        "name": "Arts",
                        "subjects": ["Arts plastiques", "Musique", "Art dramatique"]
                    }
                }
            
            # Route /api/health
            elif self.path == '/api/health':
                response = {
                    "status": "healthy", 
                    "assistant": "Étienne",
                    "platform": "vercel",
                    "mongodb": "configured" if os.environ.get('MONGO_URL') else "not_configured",
                    "huggingface": "configured" if os.environ.get('HUGGINGFACE_TOKEN') else "not_configured"
                }
            
            else:
                response = {
                    "error": "Endpoint not found", 
                    "available_endpoints": ["/api", "/api/subjects", "/api/health", "/api/chat", "/api/analyze-text"],
                    "requested_path": self.path
                }
            
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            error_response = {
                "error": "Internal server error",
                "message": str(e),
                "endpoint": self.path
            }
            self.wfile.write(json.dumps(error_response).encode('utf-8'))

    def do_POST(self):
        """Handle POST requests"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        try:
            # Lire le body de la requête
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                post_data = self.rfile.read(content_length)
                request_data = json.loads(post_data.decode('utf-8'))
            else:
                request_data = {}
            
            # Route /api/chat
            if self.path == '/api/chat':
                message = request_data.get('message', '')
                message_type = request_data.get('message_type', 'je_veux')
                session_id = request_data.get('session_id', 'web-session')
                
                # Sources anglaises spécialisées avec nouvelles sources
                english_sources_data = {
                    "literature": [
                        "CliffsNotes Literature Guides (cliffsnotes.com)",
                        "SparkNotes Literature (sparknotes.com)",
                        "Lecturia Academic Library (lecturia.com)",
                        "Project Gutenberg (gutenberg.org)"
                    ],
                    "grammar": [
                        "Oxford English Grammar (oxford.com)",
                        "Cambridge Grammar (cambridge.org)",
                        "Purdue OWL Writing Lab (owl.purdue.edu)",
                        "BBC Learning English (bbc.co.uk/learningenglish)"
                    ],
                    "academic": [
                        "Purdue OWL Writing Lab (owl.purdue.edu)",
                        "Harvard Writing Center (writingcenter.fas.harvard.edu)",
                        "McGill Writing Centre (mcgill.ca/mwc)",
                        "CliffsNotes Study Guides (cliffsnotes.com/study-guides)"
                    ]
                }
                
                message_lower = message.lower()
                
                # Détection de langue
                english_words = ["the", "and", "help", "me", "grammar", "writing", "english"]
                french_words = ["le", "de", "aide", "moi", "grammaire", "français"]
                
                english_count = sum(1 for word in english_words if word in message_lower)
                french_count = sum(1 for word in french_words if word in message_lower)
                detected_lang = "en" if english_count > french_count and english_count > 0 else "fr"
                
                # Réponses selon la langue et le contenu
                if any(word in message_lower for word in ["english", "shakespeare", "literature", "grammar"]):
                    if "shakespeare" in message_lower or "literature" in message_lower:
                        if detected_lang == "en":
                            response_text = f"For English literature about '{message}', I recommend these excellent academic sources:\n\n📚 **Literature Sources:**\n1. CliffsNotes Literature Guides - Comprehensive analysis\n2. SparkNotes Literature - Detailed summaries and themes\n3. Lecturia Academic Library - Scholarly articles\n4. Project Gutenberg - Original texts\n\n💡 These sources provide thorough academic analysis perfect for students."
                        else:
                            response_text = f"Pour la littérature anglaise concernant '{message}', voici d'excellentes sources académiques:\n\n📚 **Sources Littéraires:**\n1. CliffsNotes Literature Guides - Analyses complètes\n2. SparkNotes Literature - Résumés détaillés\n3. Lecturia Academic Library - Articles savants\n4. Project Gutenberg - Textes originaux\n\n💡 Ces sources offrent des analyses approfondies parfaites pour les étudiants."
                        sources = english_sources_data["literature"]
                        trust_score = 0.95
                    else:
                        if detected_lang == "en":
                            response_text = f"For English grammar regarding '{message}', here are the gold standard resources:\n\n📝 **Grammar Sources:**\n1. Oxford English Grammar - The definitive reference\n2. Cambridge Grammar - Academic standard\n3. Purdue OWL Writing Lab - University writing guide\n4. BBC Learning English - Practical lessons\n\n💡 These are internationally recognized for English language learning."
                        else:
                            response_text = f"Pour la grammaire anglaise concernant '{message}', voici les références de référence:\n\n📝 **Sources Grammaire:**\n1. Oxford English Grammar - La référence mondiale\n2. Cambridge Grammar - Standard académique\n3. Purdue OWL Writing Lab - Guide universitaire\n4. BBC Learning English - Leçons pratiques\n\n💡 Ces sources sont reconnues internationalement."
                        sources = english_sources_data["grammar"]
                        trust_score = 0.95
                else:
                    # Réponse française avec sources québécoises
                    response_text = f"Bonjour ! Je suis **Étienne**, votre assistant éducatif. 🎓\n\nConcernant '{message}': Pour cette question d'étudiant québécois, voici mes recommandations:\n\n🍁 **Sources Québécoises Officielles:**\n1. Sites gouvernementaux (.gouv.qc.ca)\n2. Ressources éducatives du MEES\n3. Universités québécoises\n4. BANQ (Bibliothèque nationale du Québec)\n\n💡 Ces sources garantissent la qualité selon le programme éducatif québécois."
                    sources = ["Sources éducatives québécoises officielles", "Sites gouvernementaux (.gouv.qc.ca)", "Universités québécoises"]
                    trust_score = 0.85
                
                # Ajout selon le type de message
                type_additions = {
                    "je_veux": "\n\n🎯 **Conseil d'Étienne:** Créez un plan d'étude structuré avec objectifs clairs et échéanciers.",
                    "je_recherche": "\n\n🔍 **Méthode de recherche:** 1) Définissez vos mots-clés 2) Croisez les sources fiables 3) Citez correctement.",
                    "sources_fiables": "\n\n✅ **Validation des sources:** Privilégiez .edu, .gouv, et institutions reconnues.",
                    "activites": "\n\n📋 **Idée d'activité:** Créez un projet avec bibliographie, présentation et réflexion critique."
                }
                
                if message_type in type_additions:
                    response_text += type_additions[message_type]
                
                response = {
                    "id": f"msg-{hash(message) % 10000}",
                    "session_id": session_id,
                    "message": message,
                    "response": response_text,
                    "message_type": message_type,
                    "trust_score": trust_score,
                    "sources": sources,
                    "timestamp": "2024-10-10T20:30:00Z",
                    "detected_language": detected_lang
                }
            
            # Route /api/analyze-text
            elif self.path == '/api/analyze-text':
                text = request_data.get('text', '')
                
                # Analyse IA améliorée
                ai_indicators = [
                    "as an ai", "i'm an ai", "as a language model", 
                    "however", "furthermore", "moreover", "in conclusion", 
                    "to summarize", "it's important to note"
                ]
                
                text_lower = text.lower()
                ai_score = 0
                detected_patterns = []
                
                for indicator in ai_indicators:
                    if indicator in text_lower:
                        ai_score += 0.2
                        detected_patterns.append(indicator)
                
                # Analyse structure (phrases uniformes)
                sentences = text.split('.')
                if len(sentences) > 3:
                    avg_length = sum(len(s.strip().split()) for s in sentences if s.strip()) / len([s for s in sentences if s.strip()])
                    if avg_length > 15:  # Phrases très longues = suspect
                        ai_score += 0.15
                
                ai_probability = min(ai_score, 0.99)
                
                # Analyse plagiat améliorée
                words = text_lower.split()
                unique_words = set(words)
                vocabulary_diversity = len(unique_words) / len(words) if words else 0
                
                academic_phrases = [
                    "according to", "research shows", "studies indicate", 
                    "experts agree", "it has been proven", "data suggests"
                ]
                
                academic_score = sum(0.1 for phrase in academic_phrases if phrase in text_lower)
                plagiarism_risk = min(academic_score + max(0, 0.7 - vocabulary_diversity), 0.99)
                
                # Détection de langue
                english_count = sum(1 for word in ["the", "and", "to", "of", "a"] if word in text_lower)
                french_count = sum(1 for word in ["le", "de", "et", "à", "un"] if word in text_lower)
                detected_language = "en" if english_count > french_count and english_count > 0 else "fr"
                
                response = {
                    "text_length": len(text),
                    "detected_language": detected_language,
                    "ai_detection": {
                        "ai_probability": round(ai_probability, 2),
                        "is_likely_ai": ai_probability > 0.5,
                        "confidence": "High" if ai_probability > 0.7 else "Medium" if ai_probability > 0.3 else "Low",
                        "detected_patterns": detected_patterns[:3]
                    },
                    "plagiarism_check": {
                        "plagiarism_risk": round(plagiarism_risk, 2),
                        "is_suspicious": plagiarism_risk > 0.4,
                        "vocabulary_diversity": round(vocabulary_diversity, 2),
                        "risk_level": "High" if plagiarism_risk > 0.6 else "Medium" if plagiarism_risk > 0.3 else "Low",
                        "recommendation": "Vérification approfondie recommandée" if plagiarism_risk > 0.4 else "Contenu semble original"
                    },
                    "overall_assessment": {
                        "is_authentic": ai_probability < 0.5 and plagiarism_risk < 0.4,
                        "confidence_score": round((1 - ai_probability + (1 - plagiarism_risk)) / 2, 2)
                    }
                }
            
            else:
                response = {
                    "error": "GET endpoint not found", 
                    "available": ["/api", "/api/subjects", "/api/health"],
                    "requested": self.path
                }
            
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            error_response = {
                "error": "Server error in GET",
                "message": str(e)
            }
            self.wfile.write(json.dumps(error_response).encode('utf-8'))

    def do_POST(self):
        """Handle POST requests"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        try:
            # Lire le body de la requête
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                post_data = self.rfile.read(content_length)
                request_data = json.loads(post_data.decode('utf-8'))
            else:
                request_data = {}
            
            # Route /api/chat
            if self.path == '/api/chat':
                message = request_data.get('message', '')
                message_type = request_data.get('message_type', 'je_veux')
                session_id = request_data.get('session_id', 'web-session')
                
                if not message:
                    response = {
                        "error": "Message is required",
                        "received_data": request_data
                    }
                else:
                    # Sources anglaises spécialisées (NOUVELLES SOURCES AJOUTÉES)
                    english_sources = {
                        "literature": [
                            "CliffsNotes Literature Guides (cliffsnotes.com)",
                            "SparkNotes Literature (sparknotes.com)",  
                            "Lecturia Academic Library (lecturia.com)",
                            "Project Gutenberg (gutenberg.org)"
                        ],
                        "grammar": [
                            "Oxford English Grammar (oxford.com)",
                            "Cambridge Grammar (cambridge.org)",
                            "Purdue OWL Writing Lab (owl.purdue.edu)",
                            "BBC Learning English (bbc.co.uk/learningenglish)"
                        ],
                        "academic": [
                            "Purdue OWL Writing Lab (owl.purdue.edu)",
                            "Harvard Writing Center (writingcenter.fas.harvard.edu)",
                            "McGill Writing Centre (mcgill.ca/mwc)",
                            "CliffsNotes Study Guides (cliffsnotes.com/study-guides)"
                        ]
                    }
                    
                    message_lower = message.lower()
                    
                    # Détection langue
                    english_words = ["the", "and", "help", "me", "grammar", "writing", "english"]
                    french_words = ["le", "de", "aide", "moi", "grammaire", "français"]
                    
                    en_count = sum(1 for word in english_words if word in message_lower)
                    fr_count = sum(1 for word in french_words if word in message_lower)
                    detected_lang = "en" if en_count > fr_count and en_count > 0 else "fr"
                    
                    # Traitement selon la langue et le sujet
                    if any(word in message_lower for word in ["english", "shakespeare", "literature", "grammar", "writing"]):
                        if "shakespeare" in message_lower or "literature" in message_lower:
                            if detected_lang == "en":
                                response_text = f"For English literature about '{message}', I recommend these comprehensive academic sources:\n\n📚 **Top Literature Resources:**\n• **CliffsNotes Literature Guides** - In-depth analysis and themes\n• **SparkNotes Literature** - Plot summaries and character analysis\n• **Lecturia Academic Library** - Scholarly articles and criticism\n• **Project Gutenberg** - Free access to original texts\n\nThese resources provide thorough academic support for English literature studies."
                            else:
                                response_text = f"Pour la littérature anglaise sur '{message}', voici mes meilleures sources académiques:\n\n📚 **Ressources Littéraires Principales:**\n• **CliffsNotes Literature Guides** - Analyses détaillées\n• **SparkNotes Literature** - Résumés et analyses de personnages\n• **Lecturia Academic Library** - Articles académiques\n• **Project Gutenberg** - Accès gratuit aux textes originaux\n\nCes ressources offrent un support académique complet pour l'étude de la littérature anglaise."
                            sources = english_sources["literature"]
                            trust_score = 0.95
                        else:
                            if detected_lang == "en":
                                response_text = f"For English grammar about '{message}', here are the most reliable academic sources:\n\n✍️ **Grammar References:**\n• **Oxford English Grammar** - The world's leading authority\n• **Cambridge Grammar** - Academic excellence\n• **Purdue OWL Writing Lab** - University standard for writing\n• **BBC Learning English** - Practical learning resource\n\nThese represent the highest standards in English language education."
                            else:
                                response_text = f"Pour la grammaire anglaise concernant '{message}', voici les sources les plus fiables:\n\n✍️ **Références Grammaticales:**\n• **Oxford English Grammar** - L'autorité mondiale\n• **Cambridge Grammar** - Excellence académique\n• **Purdue OWL Writing Lab** - Standard universitaire\n• **BBC Learning English** - Ressource d'apprentissage\n\nCes sources représentent les plus hauts standards en enseignement anglais."
                            sources = english_sources["grammar"]
                            trust_score = 0.95
                    else:
                        # Réponse française standard
                        response_text = f"Bonjour ! Je suis **Étienne**, votre assistant éducatif québécois. 🎓\n\nPour votre question '{message}', voici mes recommandations adaptées au système éducatif québécois:\n\n🍁 **Sources Officielles Québécoises:**\n• Sites gouvernementaux (.gouv.qc.ca)\n• Ministère de l'Éducation (MEES)\n• Universités québécoises (UdeM, McGill, UQAM)\n• BANQ - Bibliothèque nationale\n\n💡 Ces ressources sont alignées avec le programme scolaire québécois et garantissent des informations fiables."
                        sources = ["Sources éducatives québécoises", "Sites gouvernementaux (.gouv.qc.ca)", "Universités québécoises"]
                        trust_score = 0.85
                    
                    response = {
                        "id": f"msg-{abs(hash(message)) % 10000}",
                        "session_id": session_id,
                        "message": message,
                        "response": response_text,
                        "message_type": message_type,
                        "trust_score": trust_score,
                        "sources": sources,
                        "timestamp": "2024-10-10T20:30:00Z",
                        "can_download": len(response_text) > 100
                    }
            
            # Route /api/analyze-text (pour vérification de texte)
            elif self.path == '/api/analyze-text':
                text = request_data.get('text', '')
                
                if not text:
                    response = {"error": "Text is required for analysis"}
                else:
                    # Analyse IA
                    ai_indicators = [
                        "as an ai", "i'm an ai", "as a language model", 
                        "however", "furthermore", "moreover", "in conclusion", 
                        "to summarize", "it's important to note", "i don't have personal"
                    ]
                    
                    text_lower = text.lower()
                    ai_score = 0
                    detected_patterns = []
                    
                    for indicator in ai_indicators:
                        if indicator in text_lower:
                            ai_score += 0.2
                            detected_patterns.append(indicator)
                    
                    ai_probability = min(ai_score, 0.99)
                    
                    # Analyse plagiat
                    words = text_lower.split()
                    unique_words = set(words)
                    vocabulary_diversity = len(unique_words) / len(words) if words else 0
                    
                    plagiarism_risk = max(0, 0.8 - vocabulary_diversity)
                    if len(text) > 500 and vocabulary_diversity < 0.5:
                        plagiarism_risk += 0.2
                    
                    # Détection langue
                    english_count = sum(1 for word in ["the", "and", "to", "of", "a"] if word in text_lower)
                    french_count = sum(1 for word in ["le", "de", "et", "à", "un"] if word in text_lower)
                    detected_language = "en" if english_count > french_count and english_count > 0 else "fr"
                    
                    response = {
                        "text_length": len(text),
                        "detected_language": detected_language,
                        "ai_detection": {
                            "ai_probability": round(ai_probability, 2),
                            "is_likely_ai": ai_probability > 0.5,
                            "confidence": "High" if ai_probability > 0.7 else "Medium" if ai_probability > 0.3 else "Low",
                            "detected_patterns": detected_patterns[:3]
                        },
                        "plagiarism_check": {
                            "plagiarism_risk": round(plagiarism_risk, 2),
                            "is_suspicious": plagiarism_risk > 0.4,
                            "vocabulary_diversity": round(vocabulary_diversity, 2),
                            "risk_level": "High" if plagiarism_risk > 0.6 else "Medium" if plagiarism_risk > 0.3 else "Low",
                            "recommendation": "Vérification approfondie recommandée" if plagiarism_risk > 0.4 else "Contenu semble original"
                        }
                    }
            
            else:
                response = {"error": "POST endpoint not found", "path": self.path}
            
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
            
        except json.JSONDecodeError as e:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            error_response = {
                "error": "Invalid JSON in request body",
                "message": str(e)
            }
            self.wfile.write(json.dumps(error_response).encode('utf-8'))
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            error_response = {
                "error": "Server error in POST",
                "message": str(e),
                "endpoint": self.path
            }
            self.wfile.write(json.dumps(error_response).encode('utf-8'))
