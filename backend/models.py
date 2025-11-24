"""Modèles Pydantic pour l'application Étienne"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

# ==================== AUTHENTIFICATION ====================

class UserSignup(BaseModel):
    full_name: str
    email: str
    password: str
    license_key: str

class UserLogin(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    success: bool
    token: str
    user: dict

class SignupResponse(BaseModel):
    success: bool
    message: str
    token: str
    user: dict

# ==================== LICENCES ====================

class LicenseCreate(BaseModel):
    organization_name: str
    license_key: str
    max_users: int = 10
    expiry_date: str
    notes: Optional[str] = None

class LicenseUpdate(BaseModel):
    max_users: Optional[int] = None
    expiry_date: Optional[str] = None
    is_active: Optional[bool] = None
    notes: Optional[str] = None

class LicenseResponse(BaseModel):
    success: bool
    message: str
    license: Optional[dict] = None

# ==================== UTILISATEURS ====================

class UserUpdate(BaseModel):
    is_active: Optional[bool] = None
    full_name: Optional[str] = None

class UserResponse(BaseModel):
    success: bool
    message: str
    user: Optional[dict] = None

# ==================== MOTS BLOQUÉS ====================

class BlockedWordCreate(BaseModel):
    word: str
    category: str  # "violence", "drugs", "illegal", "custom", etc.
    severity: str  # "low", "medium", "high", "critical"
    is_exception: bool = False  # Si True, le mot est autorisé malgré détection

class BlockedWordUpdate(BaseModel):
    category: Optional[str] = None
    severity: Optional[str] = None
    is_exception: Optional[bool] = None
    is_active: Optional[bool] = None

class BlockedWordResponse(BaseModel):
    success: bool
    message: str
    word: Optional[dict] = None

# ==================== CONVERSATIONS ====================

class ChatMessage(BaseModel):
    role: str  # "user" ou "assistant"
    content: str
    timestamp: Optional[str] = None

class ConversationResponse(BaseModel):
    success: bool
    conversations: List[dict]
    total: int

# ==================== STATISTIQUES ====================

class StatsResponse(BaseModel):
    success: bool
    stats: dict

# ==================== ADMIN ====================

class AdminLoginRequest(BaseModel):
    username: str
    password: str

class AdminLoginResponse(BaseModel):
    success: bool
    token: str
    message: str
