"""
Fonctionnalités de sécurité avancées pour Étienne
- Authentification admin
- Alertes email
- Rate limiting
- Export de rapports
"""

import os
import smtplib
import hashlib
import csv
import io
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict
from jose import JWTError, jwt
from fastapi import HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from collections import defaultdict
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch
import logging
from pathlib import Path
from dotenv import load_dotenv

# Charger .env
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Configuration depuis .env
SMTP_HOST = os.environ.get('SMTP_HOST', '')
SMTP_PORT = int(os.environ.get('SMTP_PORT', 587))
SMTP_USERNAME = os.environ.get('SMTP_USERNAME', '')
SMTP_PASSWORD = os.environ.get('SMTP_PASSWORD', '')
SMTP_FROM_EMAIL = os.environ.get('SMTP_FROM_EMAIL', '')
SECURITY_ALERT_EMAIL = os.environ.get('SECURITY_ALERT_EMAIL', '')

ADMIN_API_TOKEN = os.environ.get('ADMIN_API_TOKEN', '')
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'default_secret_key')
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admetienne')
ADMIN_PASSWORD_HASH = os.environ.get('ADMIN_PASSWORD_HASH', '')
ADMIN_EMAIL = os.environ.get('ADMIN_EMAIL', '')

# Rate Limiting Storage (en mémoire - pour production utiliser Redis)
rate_limit_storage: Dict[str, list] = defaultdict(list)
RATE_LIMIT_ATTEMPTS = 10  # tentatives max
RATE_LIMIT_WINDOW = 60  # secondes

# Security Bearer
security = HTTPBearer()

# ============================================
# AUTHENTIFICATION ADMIN
# ============================================

def hash_password(password: str) -> str:
    """Hash un mot de passe avec SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """Vérifie un mot de passe"""
    return hash_password(password) == hashed

def create_access_token(data: dict) -> str:
    """Crée un JWT token"""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> dict:
    """Vérifie un JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Token invalide ou expiré")

def verify_api_token(token: str) -> bool:
    """Vérifie le token API admin"""
    return token == ADMIN_API_TOKEN

async def get_current_admin(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Dépendance pour vérifier l'authentification admin (JWT ou API Token)"""
    token = credentials.credentials
    
    # Essayer API Token d'abord
    if verify_api_token(token):
        return {"username": "admin_api", "type": "api_token"}
    
    # Sinon, vérifier JWT
    try:
        payload = verify_token(token)
        if payload.get("username") != ADMIN_USERNAME:
            raise HTTPException(status_code=403, detail="Accès refusé")
        return {"username": payload["username"], "type": "jwt"}
    except:
        raise HTTPException(status_code=401, detail="Authentification requise")

# ============================================
# RATE LIMITING PAR IP
# ============================================

def check_rate_limit(ip_address: str) -> bool:
    """
    Vérifie le rate limiting pour une IP
    Retourne True si l'IP est dans la limite, False si bloquée
    """
    current_time = datetime.now(timezone.utc)
    
    # Nettoyer les anciennes tentatives
    rate_limit_storage[ip_address] = [
        attempt_time for attempt_time in rate_limit_storage[ip_address]
        if (current_time - attempt_time).total_seconds() < RATE_LIMIT_WINDOW
    ]
    
    # Vérifier le nombre de tentatives
    if len(rate_limit_storage[ip_address]) >= RATE_LIMIT_ATTEMPTS:
        return False
    
    # Ajouter la tentative actuelle
    rate_limit_storage[ip_address].append(current_time)
    return True

def get_client_ip(request: Request) -> str:
    """Extrait l'IP du client depuis les headers"""
    # Vérifier les headers de proxy
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback sur l'IP directe
    if request.client:
        return request.client.host
    
    return "unknown"

# ============================================
# ALERTES EMAIL
# ============================================

def send_email_alert(subject: str, body: str, attachments: list = None) -> bool:
    """
    Envoie une alerte email via SMTP2GO
    """
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_FROM_EMAIL
        msg['To'] = SECURITY_ALERT_EMAIL
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'html'))
        
        # Ajouter les pièces jointes si présentes
        if attachments:
            for filename, content in attachments:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(content)
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename={filename}')
                msg.attach(part)
        
        # Connexion SMTP
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        
        logging.info(f"Email d'alerte envoyé: {subject}")
        return True
        
    except Exception as e:
        logging.error(f"Erreur envoi email: {e}")
        return False

def send_critical_alert(log_data: dict) -> bool:
    """Envoie une alerte immédiate pour tentative critique"""
    subject = f"🚨 ALERTE SÉCURITÉ CRITIQUE - {log_data['detected_category'].upper()}"
    
    body = f"""
    <html>
    <body style="font-family: Arial, sans-serif;">
        <h2 style="color: #dc2626;">🚨 Alerte Sécurité Critique</h2>
        
        <div style="background-color: #fee2e2; padding: 15px; border-left: 4px solid #dc2626; margin: 20px 0;">
            <p><strong>Catégorie:</strong> {log_data['detected_category']}</p>
            <p><strong>Sévérité:</strong> <span style="color: #dc2626; font-weight: bold;">{log_data['severity'].upper()}</span></p>
            <p><strong>Date/Heure:</strong> {log_data['timestamp']}</p>
        </div>
        
        <h3>Détails de la tentative:</h3>
        <ul>
            <li><strong>Session ID:</strong> {log_data.get('session_id', 'N/A')}</li>
            <li><strong>IP Address:</strong> {log_data.get('ip_address', 'N/A')}</li>
            <li><strong>Message:</strong> {log_data['user_message']}</li>
            <li><strong>Mots-clés détectés:</strong> {', '.join(log_data['keywords_matched'])}</li>
        </ul>
        
        <hr style="margin: 20px 0; border: 1px solid #e5e7eb;">
        
        <p style="color: #6b7280; font-size: 12px;">
            Ceci est une alerte automatique du système de sécurité Étienne.
            <br>Collège Champagneur - Système de surveillance IA
        </p>
    </body>
    </html>
    """
    
    return send_email_alert(subject, body)

# ============================================
# EXPORT CSV/PDF
# ============================================

def generate_csv_report(logs: list) -> bytes:
    """Génère un rapport CSV des logs"""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Headers
    writer.writerow([
        'ID', 'Date/Heure', 'Session ID', 'IP Address', 
        'Catégorie', 'Sévérité', 'Message', 'Mots-clés'
    ])
    
    # Données
    for log in logs:
        writer.writerow([
            log.get('id', ''),
            log.get('timestamp', ''),
            log.get('session_id', ''),
            log.get('ip_address', 'N/A'),
            log.get('detected_category', ''),
            log.get('severity', ''),
            log.get('user_message', '')[:100],
            ', '.join(log.get('keywords_matched', []))
        ])
    
    return output.getvalue().encode('utf-8')

def generate_pdf_report(logs: list, stats: dict) -> bytes:
    """Génère un rapport PDF des logs"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    # Titre
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#dc2626'),
        spaceAfter=30
    )
    elements.append(Paragraph("🚨 Rapport de Sécurité Étienne", title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Statistiques
    stats_style = ParagraphStyle(
        'Stats',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=10
    )
    elements.append(Paragraph(f"<b>Date du rapport:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC", stats_style))
    elements.append(Paragraph(f"<b>Total de tentatives:</b> {stats.get('total_logs', 0)}", stats_style))
    elements.append(Paragraph(f"<b>Tentatives critiques:</b> {stats.get('critical_count', 0)}", stats_style))
    elements.append(Paragraph(f"<b>Tentatives à risque élevé:</b> {stats.get('high_count', 0)}", stats_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Catégories
    if stats.get('category_stats'):
        elements.append(Paragraph("<b>Répartition par catégorie:</b>", stats_style))
        for category, count in stats['category_stats'].items():
            elements.append(Paragraph(f"  • {category}: {count}", stats_style))
        elements.append(Spacer(1, 0.3*inch))
    
    # Tableau des logs (derniers 20)
    elements.append(Paragraph("<b>Détails des tentatives (20 dernières):</b>", title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    table_data = [['Date', 'IP', 'Catégorie', 'Sévérité', 'Message']]
    
    for log in logs[:20]:
        table_data.append([
            str(log.get('timestamp', ''))[:19],
            log.get('ip_address', 'N/A')[:15],
            log.get('detected_category', '')[:15],
            log.get('severity', ''),
            log.get('user_message', '')[:40] + '...'
        ])
    
    table = Table(table_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 0.8*inch, 2.3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc2626')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')])
    ]))
    
    elements.append(table)
    
    # Footer
    elements.append(Spacer(1, 0.5*inch))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey
    )
    elements.append(Paragraph(
        "Rapport généré automatiquement par le système de sécurité Étienne - Collège Champagneur",
        footer_style
    ))
    
    doc.build(elements)
    return buffer.getvalue()

def send_daily_report(logs: list, stats: dict) -> bool:
    """Envoie un rapport quotidien par email"""
    subject = f"📊 Rapport Quotidien de Sécurité - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    
    body = f"""
    <html>
    <body style="font-family: Arial, sans-serif;">
        <h2 style="color: #2563eb;">📊 Rapport Quotidien de Sécurité</h2>
        <p><strong>Date:</strong> {datetime.now(timezone.utc).strftime('%Y-%m-%d')}</p>
        
        <h3>Résumé:</h3>
        <ul>
            <li><strong>Total de tentatives:</strong> {stats.get('total_logs', 0)}</li>
            <li><strong>Tentatives critiques:</strong> {stats.get('critical_count', 0)}</li>
            <li><strong>Tentatives à risque élevé:</strong> {stats.get('high_count', 0)}</li>
        </ul>
        
        <p>Les rapports détaillés (CSV et PDF) sont joints à cet email.</p>
        
        <hr>
        <p style="color: #6b7280; font-size: 12px;">
            Rapport automatique du système de sécurité Étienne - Collège Champagneur
        </p>
    </body>
    </html>
    """
    
    # Générer les fichiers
    csv_content = generate_csv_report(logs)
    pdf_content = generate_pdf_report(logs, stats)
    
    attachments = [
        (f"security_report_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv", csv_content),
        (f"security_report_{datetime.now(timezone.utc).strftime('%Y%m%d')}.pdf", pdf_content)
    ]
    
    return send_email_alert(subject, body, attachments)
