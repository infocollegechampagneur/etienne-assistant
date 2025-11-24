"""Routes d'administration pour Étienne"""

from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime, timezone
import uuid
import logging

from models import (
    LicenseCreate, LicenseUpdate, LicenseResponse,
    UserUpdate, UserResponse,
    BlockedWordCreate, BlockedWordUpdate, BlockedWordResponse,
    StatsResponse, ConversationResponse
)
from security_advanced import get_current_admin
from utils import conversation_store

# Router
router = APIRouter(prefix="/admin", tags=["Administration"])

# MongoDB connection (sera injecté depuis server.py)
db = None

def set_db(database):
    """Configure la connexion à la base de données"""
    global db
    db = database

# ==================== STATISTIQUES ====================

@router.get("/stats", response_model=StatsResponse)
async def get_stats(admin = Depends(get_current_admin)):
    """Obtenir les statistiques globales (Admin seulement)"""
    try:
        # Compter les licences
        total_licenses = await db.licenses.count_documents({})
        active_licenses = await db.licenses.count_documents({"is_active": True})
        
        # Compter les licences expirées
        all_licenses = await db.licenses.find({}, {"_id": 0, "expiry_date": 1, "is_active": 1}).to_list(1000)
        expired_count = 0
        for lic in all_licenses:
            if lic.get("is_active", True):
                expiry_str = lic["expiry_date"]
                if 'T' in expiry_str:
                    expiry = datetime.fromisoformat(expiry_str)
                else:
                    expiry = datetime.strptime(expiry_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
                
                if expiry.tzinfo is None:
                    expiry = expiry.replace(tzinfo=timezone.utc)
                
                if expiry < datetime.now(timezone.utc):
                    expired_count += 1
        
        # Compter les utilisateurs
        total_users = await db.users.count_documents({})
        active_users = await db.users.count_documents({"is_active": True})
        
        # Compter les mots bloqués
        blocked_words_count = await db.blocked_words.count_documents({})
        
        # Stats des conversations (depuis le store en mémoire)
        conversation_stats = conversation_store.get_stats()
        
        return StatsResponse(
            success=True,
            stats={
                "licenses": {
                    "total": total_licenses,
                    "active": active_licenses,
                    "expired": expired_count
                },
                "users": {
                    "total": total_users,
                    "active": active_users
                },
                "blocked_words": blocked_words_count,
                "messages_processed": conversation_stats["total_messages_stored"]
            }
        )
    except Exception as e:
        logging.error(f"Erreur stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== GESTION DES LICENCES ====================

@router.get("/licenses")
async def get_all_licenses(admin = Depends(get_current_admin)):
    """Obtenir toutes les licences avec statistiques (Admin seulement)"""
    try:
        licenses = await db.licenses.find({}, {"_id": 0}).to_list(1000)
        
        # Ajouter les stats pour chaque licence
        for license in licenses:
            user_count = await db.users.count_documents({"license_key": license["license_key"]})
            license["current_users"] = user_count
            
            # Parser la date d'expiration avec gestion des timezones
            expiry_date_str = license["expiry_date"]
            if 'T' in expiry_date_str:
                expiry_date = datetime.fromisoformat(expiry_date_str)
            else:
                expiry_date = datetime.strptime(expiry_date_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
            
            if expiry_date.tzinfo is None:
                expiry_date = expiry_date.replace(tzinfo=timezone.utc)
            
            license["is_expired"] = expiry_date < datetime.now(timezone.utc)
        
        return {
            "success": True,
            "licenses": licenses,
            "total_licenses": len(licenses)
        }
        
    except Exception as e:
        logging.error(f"Erreur récupération licences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/licenses", response_model=LicenseResponse)
async def create_license(license_data: LicenseCreate, admin = Depends(get_current_admin)):
    """Créer une nouvelle licence (Admin seulement)"""
    try:
        # Vérifier si la clé existe déjà
        existing = await db.licenses.find_one({"license_key": license_data.license_key})
        if existing:
            raise HTTPException(status_code=400, detail="Cette clé de licence existe déjà")
        
        license_doc = {
            "id": str(uuid.uuid4()),
            "organization_name": license_data.organization_name,
            "license_key": license_data.license_key,
            "max_users": license_data.max_users,
            "expiry_date": license_data.expiry_date,
            "is_active": True,
            "notes": license_data.notes or "",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": admin["user_id"]
        }
        
        await db.licenses.insert_one(license_doc)
        
        return LicenseResponse(
            success=True,
            message="Licence créée avec succès",
            license=license_doc
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Erreur création licence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/licenses/{license_key}", response_model=LicenseResponse)
async def update_license(license_key: str, update_data: LicenseUpdate, admin = Depends(get_current_admin)):
    """Mettre à jour une licence (Admin seulement)"""
    try:
        license_doc = await db.licenses.find_one({"license_key": license_key})
        if not license_doc:
            raise HTTPException(status_code=404, detail="Licence non trouvée")
        
        update_fields = {}
        if update_data.max_users is not None:
            update_fields["max_users"] = update_data.max_users
        if update_data.expiry_date is not None:
            update_fields["expiry_date"] = update_data.expiry_date
        if update_data.is_active is not None:
            update_fields["is_active"] = update_data.is_active
        if update_data.notes is not None:
            update_fields["notes"] = update_data.notes
        
        update_fields["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        await db.licenses.update_one(
            {"license_key": license_key},
            {"$set": update_fields}
        )
        
        return LicenseResponse(
            success=True,
            message="Licence mise à jour avec succès"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Erreur mise à jour licence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/licenses/{license_key}")
async def delete_license(license_key: str, admin = Depends(get_current_admin)):
    """Supprimer une licence (Admin seulement)"""
    try:
        # Vérifier si des utilisateurs utilisent cette licence
        user_count = await db.users.count_documents({"license_key": license_key})
        if user_count > 0:
            raise HTTPException(
                status_code=400, 
                detail=f"Impossible de supprimer: {user_count} utilisateur(s) actif(s)"
            )
        
        result = await db.licenses.delete_one({"license_key": license_key})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Licence non trouvée")
        
        return {"success": True, "message": "Licence supprimée"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Erreur suppression licence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/licenses/{license_key}/users")
async def get_license_users(license_key: str, admin = Depends(get_current_admin)):
    """Obtenir les utilisateurs d'une licence (Admin seulement)"""
    try:
        users = await db.users.find(
            {"license_key": license_key},
            {"_id": 0, "password": 0}
        ).to_list(1000)
        
        return {
            "success": True,
            "license_key": license_key,
            "users": users,
            "total_users": len(users)
        }
    except Exception as e:
        logging.error(f"Erreur récupération utilisateurs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== GESTION DES UTILISATEURS ====================

@router.get("/users")
async def get_all_users(admin = Depends(get_current_admin)):
    """Obtenir tous les utilisateurs (Admin seulement)"""
    try:
        users = await db.users.find({}, {"_id": 0, "password": 0}).to_list(1000)
        
        # Ajouter le nombre de conversations pour chaque utilisateur
        for user in users:
            user["conversation_count"] = conversation_store.get_conversation_count(user["id"])
        
        return {
            "success": True,
            "users": users,
            "total_users": len(users)
        }
    except Exception as e:
        logging.error(f"Erreur récupération utilisateurs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(user_id: str, update_data: UserUpdate, admin = Depends(get_current_admin)):
    """Mettre à jour un utilisateur (Admin seulement)"""
    try:
        user = await db.users.find_one({"id": user_id})
        if not user:
            raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
        
        update_fields = {}
        if update_data.is_active is not None:
            update_fields["is_active"] = update_data.is_active
        if update_data.full_name is not None:
            update_fields["full_name"] = update_data.full_name
        
        update_fields["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        await db.users.update_one(
            {"id": user_id},
            {"$set": update_fields}
        )
        
        return UserResponse(
            success=True,
            message="Utilisateur mis à jour avec succès"
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Erreur mise à jour utilisateur: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/users/{user_id}")
async def delete_user(user_id: str, admin = Depends(get_current_admin)):
    """Supprimer un utilisateur (Admin seulement)"""
    try:
        result = await db.users.delete_one({"id": user_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
        
        # Supprimer aussi ses conversations
        conversation_store.clear_user_conversations(user_id)
        
        return {"success": True, "message": "Utilisateur supprimé"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Erreur suppression utilisateur: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users/{user_id}/conversations", response_model=ConversationResponse)
async def get_user_conversations(user_id: str, admin = Depends(get_current_admin)):
    """Obtenir l'historique des conversations d'un utilisateur (Admin seulement)"""
    try:
        # Vérifier que l'utilisateur existe
        user = await db.users.find_one({"id": user_id}, {"_id": 0, "full_name": 1, "email": 1})
        if not user:
            raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
        
        conversations = conversation_store.get_user_conversations(user_id, limit=200)
        
        return ConversationResponse(
            success=True,
            conversations=conversations,
            total=len(conversations)
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Erreur récupération conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== GESTION DES MOTS BLOQUÉS ====================

@router.get("/blocked-words")
async def get_blocked_words(admin = Depends(get_current_admin)):
    """Obtenir tous les mots bloqués (Admin seulement)"""
    try:
        words = await db.blocked_words.find({}, {"_id": 0}).to_list(1000)
        
        return {
            "success": True,
            "words": words,
            "total_words": len(words)
        }
    except Exception as e:
        logging.error(f"Erreur récupération mots bloqués: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/blocked-words", response_model=BlockedWordResponse)
async def create_blocked_word(word_data: BlockedWordCreate, admin = Depends(get_current_admin)):
    """Ajouter un mot bloqué (Admin seulement)"""
    try:
        # Vérifier si le mot existe déjà
        existing = await db.blocked_words.find_one({"word": word_data.word.lower()})
        if existing:
            raise HTTPException(status_code=400, detail="Ce mot existe déjà dans la liste")
        
        word_doc = {
            "id": str(uuid.uuid4()),
            "word": word_data.word.lower(),
            "category": word_data.category,
            "severity": word_data.severity,
            "is_exception": word_data.is_exception,
            "is_active": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": admin["user_id"]
        }
        
        await db.blocked_words.insert_one(word_doc)
        
        return BlockedWordResponse(
            success=True,
            message="Mot ajouté avec succès",
            word=word_doc
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Erreur ajout mot bloqué: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/blocked-words/{word_id}", response_model=BlockedWordResponse)
async def update_blocked_word(word_id: str, update_data: BlockedWordUpdate, admin = Depends(get_current_admin)):
    """Mettre à jour un mot bloqué (Admin seulement)"""
    try:
        word = await db.blocked_words.find_one({"id": word_id})
        if not word:
            raise HTTPException(status_code=404, detail="Mot non trouvé")
        
        update_fields = {}
        if update_data.category is not None:
            update_fields["category"] = update_data.category
        if update_data.severity is not None:
            update_fields["severity"] = update_data.severity
        if update_data.is_exception is not None:
            update_fields["is_exception"] = update_data.is_exception
        if update_data.is_active is not None:
            update_fields["is_active"] = update_data.is_active
        
        update_fields["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        await db.blocked_words.update_one(
            {"id": word_id},
            {"$set": update_fields}
        )
        
        return BlockedWordResponse(
            success=True,
            message="Mot mis à jour avec succès"
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Erreur mise à jour mot bloqué: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/blocked-words/{word_id}")
async def delete_blocked_word(word_id: str, admin = Depends(get_current_admin)):
    """Supprimer un mot bloqué (Admin seulement)"""
    try:
        result = await db.blocked_words.delete_one({"id": word_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Mot non trouvé")
        
        return {"success": True, "message": "Mot supprimé"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Erreur suppression mot bloqué: {e}")
        raise HTTPException(status_code=500, detail=str(e))
