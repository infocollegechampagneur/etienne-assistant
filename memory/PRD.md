# PRD - Étienne: Assistant IA pour Enseignants du Québec

## Vision Produit
Application SaaS multi-tenant permettant aux enseignants du secondaire québécois d'utiliser un assistant IA pour la préparation de cours, la correction et l'analyse de textes.

---

## Fonctionnalités Implémentées ✅

### 1. Système d'Authentification & Licences
- ✅ Inscription avec clé de licence
- ✅ Connexion/Déconnexion utilisateur
- ✅ Gestion des licences par Super Admin
- ✅ **NOUVEAU: Réinitialisation de mot de passe par email (SMTP2GO)**

### 2. Système Admin Multi-Niveaux
| Rôle | Permissions |
|------|-------------|
| **Super Admin** | Tout gérer: licences, utilisateurs, prolongations, désigner admins de licence |
| **Admin de Licence** | Modifier clé de licence, ajouter/retirer utilisateurs (limite fixée par Super Admin) |
| **Utilisateur** | Utiliser l'application |

### 3. Conversations Cloud Synchronisées (NOUVEAU - 30 Jan 2025)
- ✅ **Historique synchronisé**: Les conversations sont stockées dans MongoDB
- ✅ **Accès multi-appareils**: Connectez-vous depuis n'importe quel ordinateur et retrouvez vos conversations
- ✅ **Mémoire de conversation**: Étienne se souvient du contexte et peut continuer la discussion sans répéter
- ✅ Fonctionnalités: épingler, favoris, renommer, exporter en TXT, supprimer

### 4. Réinitialisation de Mot de Passe (NOUVEAU - 30 Jan 2025)
- ✅ Lien "Mot de passe oublié?" dans le formulaire de connexion
- ✅ Email de réinitialisation envoyé via SMTP2GO
- ✅ Lien valide 1 heure
- ✅ Interface de création de nouveau mot de passe

### 5. Assistant IA Bilingue avec Mémoire
- ✅ Étienne (français) / Steven (anglais)
- ✅ Détection automatique de la langue
- ✅ Branding "Collège Champagneur" (ne mentionne jamais Google/Gemini)
- ✅ **NOUVEAU: Mémoire de conversation** - Étienne se souvient des échanges précédents

### 6. Fonctionnalités Enseignants
- ✅ Chat avec IA (plans de cours, exercices, etc.)
- ✅ Analyse de fichiers (PDF, Word, TXT, images)
- ✅ Génération de graphiques mathématiques
- ✅ Génération d'images artistiques (Hugging Face - requiert clé API)
- ✅ Export PDF/Word des réponses

### 7. Support Mathématiques LaTeX (NOUVEAU - 30 Jan 2025)
- ✅ **Formules LaTeX** dans les réponses ($..$ pour inline, $$...$$ pour blocs)
- ✅ **Rendu KaTeX** dans le frontend (react-katex)
- ✅ Symboles : fractions ($\frac{a}{b}$), vecteurs ($\vec{AB}$), racines ($\sqrt{x}$), exposants, indices, etc.

### 8. Design Exports Amélioré (NOUVEAU - 30 Jan 2025)
- ✅ **Logo Champagneur** intégré dans les exports PDF et Word
- ✅ Couleurs branding rouge foncé (#8B0000) et gris
- ✅ En-têtes professionnels avec nom de l'école

### 9. Contraintes MELS Renforcées (NOUVEAU - 30 Jan 2025)
- ✅ **Strict respect du curriculum** PFEQ Sec 1-5
- ✅ **Interdiction explicite** du contenu CÉGEP/université
- ✅ Liste détaillée des concepts par niveau et séquence (CST, TS, SN)
- ✅ Instructions LaTeX pour les formules mathématiques

---

## Configuration Requise

### Variables d'Environnement Backend
```
MONGO_URL=mongodb://localhost:27017
DB_NAME=test_database
JWT_SECRET_KEY=...
GOOGLE_API_KEY=AIzaSyAU_NtQAJsdxt5AAgNTPr6bSPc3quIlRSM

# SMTP pour emails de réinitialisation (SMTP2GO)
SMTP_HOST=mail.smtp2go.com
SMTP_PORT=587
SMTP_USERNAME=securiteia@champagneur.qc.ca
SMTP_PASSWORD=ZcJpNouHAjvpgC8G
SMTP_FROM_EMAIL=securiteia@champagneur.qc.ca

# Optionnel
HUGGINGFACE_API_KEY=... (pour images artistiques)
ADMIN_EMAIL=informatique@champagneur.qc.ca
```

---

## Fichiers Modifiés/Ajoutés (Session 30 Jan 2025)

### Nouveaux Fichiers
| Fichier | Description |
|---------|-------------|
| `frontend/src/ConversationService.js` | Service pour synchronisation conversations cloud |
| `frontend/src/components/MathRenderer.js` | Composant React pour rendu LaTeX |
| `backend/logo_champagneur.jpg` | Logo du Collège Champagneur (3.1MB) |
| `backend/tests/test_etienne_api.py` | Tests API automatisés |

### Fichiers Modifiés
| Fichier | Modifications |
|---------|---------------|
| `backend/server.py` | +Prompts MELS renforcés, +Instructions LaTeX, +Logo dans exports PDF/Word |
| `frontend/src/App.js` | +Import KaTeX, +Composant MessageContent pour rendu LaTeX |
| `frontend/src/utils/formatMessage.js` | Préservation des formules LaTeX dans le formatage |
| `frontend/src/App.css` | Styles pour KaTeX (.katex-block-container, etc.) |
| `frontend/src/ConversationSidebar.js` | Utilise ConversationService au lieu de localStorage |
| `frontend/src/components/AuthModal.js` | Ajout formulaire "Mot de passe oublié" |

---

## Backlog (P0/P1/P2)

### P0 - Critique
- [x] ~~Conversations synchronisées cloud~~
- [x] ~~Mémoire de conversation~~
- [x] ~~Réinitialisation mot de passe~~
- [x] ~~Support LaTeX pour formules mathématiques~~ ✅ (30 Jan 2025)
- [x] ~~Design exports avec logo Champagneur~~ ✅ (30 Jan 2025)
- [x] ~~Contraintes MELS renforcées~~ ✅ (30 Jan 2025)

### P1 - Important
- [x] ~~**Bug Fix**: Onglets Admin Panel fermaient le modal~~ ✅ (16 Fév 2025)
- [x] ~~**Bug Fix**: Impression LaTeX affichait le code brut~~ ✅ (16 Fév 2025)
- [ ] **Bug Fix**: Réinitialisation mot de passe crash sur Render (SMTP env vars?)
- [ ] **Refactoring majeur** du code (server.py > 3500 lignes, App.js > 1700 lignes)
- [ ] Gestion complète des utilisateurs dans Admin Panel
- [ ] Modération de contenu (mots bloqués)
- [ ] Historique des conversations admin

### P2 - Améliorations
- [ ] Statistiques avancées tableau de bord
- [ ] Notifications expiration licences
- [ ] Export des données utilisateurs

---

## Architecture Technique

```
/app
├── backend/
│   ├── server.py                  # FastAPI (~5450 lignes, avec INDEX de navigation)
│   ├── ai_detector_advanced.py    # Détecteur IA
│   ├── security_advanced.py       # JWT
│   ├── assets/
│   │   └── logo_champagneur.jpg
│   ├── tests/
│   │   └── test_etienne_api.py
│   ├── requirements.txt
│   └── .env
├── frontend/
│   ├── src/
│   │   ├── App.js                 # Composant principal (~1700 lignes)
│   │   ├── AdminPanel.js          # Panel Super Admin (corrigé: tabs bug)
│   │   ├── LicenseAdminPanel.js   # Panel Admin Licence (corrigé: tabs bug)
│   │   ├── ConversationService.js # Service cloud
│   │   ├── ConversationSidebar.js
│   │   ├── QuotaDisplay.js
│   │   ├── MathSymbolsDemo.js
│   │   ├── components/
│   │   │   ├── AuthModal.js
│   │   │   ├── MathRenderer.js
│   │   │   └── ui/               # Shadcn components
│   │   └── utils/
│   │       └── formatMessage.js
│   ├── package.json
│   └── .env
└── memory/
    └── PRD.md
│   │   └── components/
│   │       └── AuthModal.js        # Inclut reset password
│   └── package.json
```

---

## Collections MongoDB

| Collection | Description |
|------------|-------------|
| `licenses` | Licences d'organisations |
| `users` | Utilisateurs |
| `conversations` | **NOUVEAU** - Conversations synchronisées |
| `password_resets` | **NOUVEAU** - Tokens de réinitialisation |

---

## Credentials Test

**Super Admin:**
- Email: `informatique@champagneur.qc.ca`
- Password: `!0910Hi8ki8+`

---

## Changelog

### 30 Janvier 2025
- ✅ **Conversations cloud**: Historique synchronisé dans MongoDB, accessible depuis n'importe quel appareil
- ✅ **Mémoire de conversation**: Étienne se souvient du contexte (10 derniers messages)
- ✅ **Réinitialisation mot de passe**: Email via SMTP2GO avec lien sécurisé (1h de validité)
- ✅ Nouveau fichier `ConversationService.js` pour API cloud
- ✅ Modifié `AuthModal.js` avec formulaire de reset password
- ✅ Modifié `ConversationSidebar.js` pour utiliser le cloud

### 22 Janvier 2025
- ✅ Système Admin multi-niveaux (Super Admin + Admin de Licence)
- ✅ Panneau LicenseAdminPanel.js
- ✅ Boutons "Désigner Admin", "Modifier Clé" dans panneau admin

### Sessions Précédentes
- Correction bug 404 upload fichiers
- Correction timeout gros PDFs
- Support bilingue Étienne/Steven
- Branding Collège Champagneur
- Génération graphiques mathématiques
- Génération images Hugging Face
