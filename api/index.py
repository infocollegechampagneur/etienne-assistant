from http.server import BaseHTTPRequestHandler
import json
import urllib.parse
import os

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        """Handle GET requests"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        # Route /api
        if self.path == '/api' or self.path == '/api/':
            response = {
                "message": "Étienne API - Assistant IA Éducatif", 
                "status": "active",
                "platform": "vercel",
                "backend": "working"
            }
        
        # Route /api/subjects
        elif self.path == '/api/subjects':
            response = {
                "langues": {"name": "Langues", "subjects": ["Français", "Anglais"]},
                "sciences": {"name": "Sciences", "subjects": ["Mathématiques", "Sciences"]},
                "arts": {"name": "Arts", "subjects": ["Arts plastiques", "Musique"]}
            }
        
        # Route /api/health
        elif self.path == '/api/health':
            response = {
                "status": "healthy", 
                "assistant": "Étienne",
                "platform": "vercel"
            }
        
        else:
            response = {"error": "Endpoint not found", "available": ["/api", "/api/subjects", "/api/health"]}
        
        self.wfile.write(json.dumps(response).encode())

    def do_POST(self):
        """Handle POST requests"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        # Lire le body de la requête
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            request_data = json.loads(post_data.decode('utf-8'))
        except:
            request_data = {}
        
        # Route /api/chat
        if self.path == '/api/chat':
            message = request_data.get('message', '')
            message_type = request_data.get('message_type', 'je_veux')
            session_id = request_data.get('session_id', 'web-session')
            
            # Sources anglaises spécialisées
            message_lower = message.lower()
            
            if any(word in message_lower for word in ["english", "shakespeare", "literature", "grammar"]):
                if "shakespeare" in message_lower or "literature" in message_lower:
                    response_text = f"For English literature about '{message}', I recommend:\n\n1. CliffsNotes Literature Guides (cliffsnotes.com)\n2. SparkNotes Literature (sparknotes.com)\n3. Lecturia Academic Library (lecturia.com)\n\nThese provide comprehensive analysis for academic work."
                    sources = [
                        "CliffsNotes Literature Guides (cliffsnotes.com)",
                        "SparkNotes Literature (sparknotes.com)",
                        "Lecturia Academic Library (lecturia.com)"
                    ]
                    trust_score = 0.95
                else:
                    response_text = f"For English grammar regarding '{message}':\n\n1. Oxford English Grammar (oxford.com)\n2. Cambridge Grammar (cambridge.org)\n3. Purdue OWL Writing Lab (owl.purdue.edu)\n\nThese are the academic standards for English language learning."
                    sources = [
                        "Oxford English Grammar (oxford.com)",
                        "Cambridge Grammar (cambridge.org)", 
                        "Purdue OWL Writing Lab (owl.purdue.edu)"
                    ]
                    trust_score = 0.95
            else:
                response_text = f"Bonjour ! Je suis Étienne, votre assistant éducatif. Concernant '{message}': Pour cette question d'étudiant québécois, je recommande les sources officielles:\n\n1. Sites gouvernementaux (.gouv.qc.ca)\n2. Ressources éducatives du MEES\n3. Universités québécoises\n\nCes sources garantissent la qualité selon le programme québécois."
                sources = ["Sources éducatives québécoises officielles"]
                trust_score = 0.85
            
            response = {
                "id": f"msg-{hash(message) % 10000}",
                "session_id": session_id,
                "message": message,
                "response": response_text,
                "message_type": message_type,
                "trust_score": trust_score,
                "sources": sources,
                "timestamp": "2024-10-10T17:30:00Z"
            }
        
        # Route /api/analyze-text
        elif self.path == '/api/analyze-text':
            text = request_data.get('text', '')
            
            # Analyse IA simple
            ai_indicators = ["however", "furthermore", "in conclusion", "as an ai"]
            ai_score = sum(0.2 for indicator in ai_indicators if indicator in text.lower())
            ai_probability = min(ai_score, 0.99)
            
            # Analyse plagiat simple
            words = text.lower().split()
            unique_words = set(words)
            diversity = len(unique_words) / len(words) if words else 0
            plagiarism_risk = max(0, 0.8 - diversity)
            
            response = {
                "ai_detection": {
                    "ai_probability": round(ai_probability, 2),
                    "is_likely_ai": ai_probability > 0.5,
                    "confidence": "High" if ai_probability > 0.7 else "Low"
                },
                "plagiarism_check": {
                    "plagiarism_risk": round(plagiarism_risk, 2),
                    "is_suspicious": plagiarism_risk > 0.4,
                    "vocabulary_diversity": round(diversity, 2)
                },
                "detected_language": "en" if any(w in text.lower() for w in ["the", "and", "help"]) else "fr"
            }
        
        else:
            response = {"error": "POST endpoint not found"}
        
        self.wfile.write(json.dumps(response).encode())
