"""Stockage en mémoire des conversations utilisateurs

Puisque nous n'utilisons pas MongoDB pour les conversations,
ce module fournit un stockage en mémoire simple.

Note: Les données seront perdues au redémarrage du serveur.
Pour une solution de production, utiliser MongoDB ou Redis.
"""

from datetime import datetime, timezone
from typing import List, Dict, Optional
import logging

# Stockage en mémoire : {user_id: [{conversation_data}]}
conversations_store: Dict[str, List[Dict]] = {}

def add_message(user_id: str, role: str, content: str, session_id: Optional[str] = None) -> bool:
    """Ajoute un message à l'historique d'un utilisateur
    
    Args:
        user_id: ID de l'utilisateur
        role: 'user' ou 'assistant'
        content: Contenu du message
        session_id: ID de session optionnel
    
    Returns:
        True si succès, False sinon
    """
    try:
        if user_id not in conversations_store:
            conversations_store[user_id] = []
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id
        }
        
        conversations_store[user_id].append(message)
        logging.info(f"Message ajouté pour user {user_id}: {role}")
        return True
    except Exception as e:
        logging.error(f"Erreur ajout message: {e}")
        return False

def get_user_conversations(user_id: str, limit: int = 100) -> List[Dict]:
    """Récupère les conversations d'un utilisateur
    
    Args:
        user_id: ID de l'utilisateur
        limit: Nombre maximum de messages à retourner
    
    Returns:
        Liste des messages (les plus récents en premier)
    """
    if user_id not in conversations_store:
        return []
    
    # Retourner les messages les plus récents en premier
    messages = conversations_store[user_id][-limit:]
    return list(reversed(messages))

def get_all_user_ids() -> List[str]:
    """Retourne la liste de tous les user_id ayant des conversations"""
    return list(conversations_store.keys())

def get_conversation_count(user_id: str) -> int:
    """Retourne le nombre de messages d'un utilisateur"""
    if user_id not in conversations_store:
        return 0
    return len(conversations_store[user_id])

def clear_user_conversations(user_id: str) -> bool:
    """Supprime toutes les conversations d'un utilisateur"""
    try:
        if user_id in conversations_store:
            del conversations_store[user_id]
            logging.info(f"Conversations supprimées pour user {user_id}")
        return True
    except Exception as e:
        logging.error(f"Erreur suppression conversations: {e}")
        return False

def clear_all_conversations() -> bool:
    """Supprime toutes les conversations (DANGER)"""
    try:
        conversations_store.clear()
        logging.warning("TOUTES les conversations ont été supprimées")
        return True
    except Exception as e:
        logging.error(f"Erreur suppression toutes conversations: {e}")
        return False

def get_stats() -> Dict:
    """Retourne des statistiques sur les conversations stockées"""
    total_users = len(conversations_store)
    total_messages = sum(len(msgs) for msgs in conversations_store.values())
    
    return {
        "total_users_with_conversations": total_users,
        "total_messages_stored": total_messages,
        "storage_type": "in-memory"
    }
