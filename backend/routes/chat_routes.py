"""Routes de chat avec enregistrement des conversations"""

from fastapi import APIRouter, HTTPException, Request
from datetime import datetime, timezone
import logging

from utils import conversation_store
from security_advanced import verify_token

# Router
router = APIRouter(tags=["Chat"])

# MongoDB connection (sera injecté depuis server.py)
db = None

def set_db(database):
    """Configure la connexion à la base de données"""
    global db
    db = database

def get_user_from_token(request: Request) -> dict:
    """Extrait l'utilisateur depuis le token JWT"""
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header.split(" ")[1]
        payload = verify_token(token)
        return payload
    except Exception as e:
        logging.warning(f"Token invalide: {e}")
        return None

async def check_blocked_words(text: str) -> tuple[bool, list]:
    """Vérifie si le texte contient des mots bloqués
    
    Returns:
        (is_blocked, found_words): True si bloqué, liste des mots trouvés
    """
    try:
        # Récupérer tous les mots bloqués actifs
        blocked_words = await db.blocked_words.find(
            {"is_active": True, "is_exception": False},
            {"_id": 0, "word": 1, "severity": 1}
        ).to_list(1000)
        
        text_lower = text.lower()
        found_words = []
        
        for word_doc in blocked_words:
            word = word_doc["word"]
            if word in text_lower:
                found_words.append(word_doc)
        
        # Bloquer uniquement si des mots de sévérité high ou critical sont trouvés
        critical_words = [w for w in found_words if w.get("severity") in ["high", "critical"]]
        
        return (len(critical_words) > 0, found_words)
    except Exception as e:
        logging.error(f"Erreur vérification mots bloqués: {e}")
        return (False, [])

@router.post("/api/chat")
async def chat_endpoint(request: Request):
    """
    Endpoint de chat principal avec enregistrement des conversations
    
    Note: Cette route doit être implémentée dans server.py car elle dépend
    de la logique d'IA existante (Gemini, génération d'images, etc.)
    
    Cette fonction est un template montrant comment intégrer l'enregistrement.
    """
    try:
        # Récupérer l'utilisateur depuis le token (optionnel, fonctionne aussi sans auth)
        user_data = get_user_from_token(request)
        user_id = user_data.get("user_id") if user_data else "anonymous"
        
        # Parser le body
        body = await request.json()
        message = body.get("message", "")
        session_id = body.get("session_id")
        
        # Vérifier les mots bloqués
        is_blocked, found_words = await check_blocked_words(message)
        if is_blocked:
            return {
                "success": False,
                "error": "Votre message contient des mots inappropriés et ne peut être traité.",
                "blocked_words": [w["word"] for w in found_words if w.get("severity") in ["high", "critical"]]
            }
        
        # Enregistrer le message de l'utilisateur
        if user_id != "anonymous":
            conversation_store.add_message(
                user_id=user_id,
                role="user",
                content=message,
                session_id=session_id
            )
        
        # === ICI: La logique d'IA existante (Gemini, etc.) ===
        # Cette partie doit être conservée depuis l'ancien endpoint /api/chat
        # Pour le moment, on retourne un message placeholder
        
        ai_response = "Cette réponse doit être générée par l'IA (Gemini)"
        
        # Enregistrer la réponse de l'assistant
        if user_id != "anonymous":
            conversation_store.add_message(
                user_id=user_id,
                role="assistant",
                content=ai_response,
                session_id=session_id
            )
            
            # Incrémenter le compteur de messages de l'utilisateur
            await db.users.update_one(
                {"id": user_id},
                {"$inc": {"message_count": 1}}
            )
        
        return {
            "success": True,
            "response": ai_response,
            "session_id": session_id
        }
        
    except Exception as e:
        logging.error(f"Erreur chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
