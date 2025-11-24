"""Routes d'authentification pour Étienne"""

from fastapi import APIRouter, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timezone
import os
import uuid
import logging

from models import UserSignup, UserLogin, LoginResponse, SignupResponse
from security_advanced import hash_password, verify_password, create_access_token

# Router
router = APIRouter(prefix="/auth", tags=["Authentication"])

# MongoDB connection (sera injecté depuis server.py)
db = None

def set_db(database):
    """Configure la connexion à la base de données"""
    global db
    db = database

@router.post("/signup", response_model=SignupResponse)
async def signup(user_data: UserSignup):
    """Inscription d'un nouvel utilisateur avec clé de licence"""
    try:
        # Vérifier si l'email existe déjà
        existing_user = await db.users.find_one({"email": user_data.email})
        if existing_user:
            raise HTTPException(status_code=400, detail="Cet email est déjà utilisé")
        
        # Vérifier la validité de la licence
        license_doc = await db.licenses.find_one({"license_key": user_data.license_key})
        if not license_doc:
            raise HTTPException(status_code=400, detail="Clé de licence invalide")
        
        if not license_doc.get("is_active", True):
            raise HTTPException(status_code=400, detail="Cette licence est désactivée")
        
        # Vérifier la date d'expiration
        expiry_date_str = license_doc["expiry_date"]
        if 'T' in expiry_date_str:
            expiry_date = datetime.fromisoformat(expiry_date_str)
        else:
            expiry_date = datetime.strptime(expiry_date_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
        
        if expiry_date.tzinfo is None:
            expiry_date = expiry_date.replace(tzinfo=timezone.utc)
        
        if expiry_date < datetime.now(timezone.utc):
            raise HTTPException(status_code=400, detail="Cette licence a expiré")
        
        # Vérifier le nombre maximum d'utilisateurs
        current_users = await db.users.count_documents({"license_key": user_data.license_key})
        if current_users >= license_doc["max_users"]:
            raise HTTPException(status_code=400, detail=f"Limite d'utilisateurs atteinte pour cette licence ({license_doc['max_users']} max)")
        
        # Hasher le mot de passe
        hashed_password = hash_password(user_data.password)
        
        # Créer l'utilisateur
        user_doc = {
            "id": str(uuid.uuid4()),
            "full_name": user_data.full_name,
            "email": user_data.email,
            "password": hashed_password,
            "license_key": user_data.license_key,
            "organization": license_doc["organization_name"],
            "is_active": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_login": None,
            "message_count": 0
        }
        
        await db.users.insert_one(user_doc)
        
        # Créer un token JWT
        token = create_access_token({"user_id": user_doc["id"], "email": user_doc["email"]})
        
        return SignupResponse(
            success=True,
            message="Compte créé avec succès",
            token=token,
            user={
                "id": user_doc["id"],
                "full_name": user_doc["full_name"],
                "email": user_doc["email"],
                "organization": user_doc["organization"]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Erreur signup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/login")
async def login(credentials: UserLogin):
    """Connexion d'un utilisateur"""
    try:
        # Trouver l'utilisateur
        user = await db.users.find_one({"email": credentials.email})
        if not user:
            raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")
        
        # Vérifier le mot de passe
        if not verify_password(credentials.password, user["password"]):
            raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")
        
        # Vérifier si l'utilisateur est actif
        if not user.get("is_active", True):
            raise HTTPException(status_code=403, detail="Votre compte a été désactivé")
        
        # Vérifier la licence
        license_doc = await db.licenses.find_one({"license_key": user["license_key"]})
        if not license_doc or not license_doc.get("is_active", True):
            raise HTTPException(status_code=403, detail="Votre licence est inactive")
        
        # Vérifier l'expiration
        expiry_date_str = license_doc["expiry_date"]
        if 'T' in expiry_date_str:
            expiry_date = datetime.fromisoformat(expiry_date_str)
        else:
            expiry_date = datetime.strptime(expiry_date_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
        
        if expiry_date.tzinfo is None:
            expiry_date = expiry_date.replace(tzinfo=timezone.utc)
        
        if expiry_date < datetime.now(timezone.utc):
            raise HTTPException(status_code=403, detail="Votre licence a expiré")
        
        # Mettre à jour la dernière connexion
        await db.users.update_one(
            {"id": user["id"]},
            {"$set": {"last_login": datetime.now(timezone.utc).isoformat()}}
        )
        
        # Créer un token JWT
        token = create_access_token({"user_id": user["id"], "email": user["email"]})
        
        return {
            "success": True,
            "token": token,
            "user": {
                "id": user["id"],
                "full_name": user["full_name"],
                "email": user["email"],
                "organization": user.get("organization", "")
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Erreur login: {e}")
        raise HTTPException(status_code=500, detail=str(e))
