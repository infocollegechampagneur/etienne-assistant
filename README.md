# 🎓 Étienne - Assistant IA pour les Étudiants Québécois

Assistant d'intelligence artificielle conçu spécialement pour les étudiants du Québec, fourni par le Collège Champagneur.

## 🚀 Fonctionnalités

✅ **Chat IA Intelligent** - Claude Sonnet 4 & Gemini 2.0 Flash  
✅ **Sources Fiables** - Scores de confiance pour sources québécoises et internationales  
✅ **Génération de Documents** - PDF, DOCX, PPTX, XLSX  
✅ **Analyse de Documents** - Upload et analyse de fichiers  
✅ **Détection IA** - Identifie les textes générés par IA  
✅ **Vérification Plagiat** - Analyse l'originalité du contenu  
✅ **Support Multilingue** - Français et Anglais adaptatif  
✅ **Sources Anglais Internationales** - Oxford, Cambridge, BBC Learning, Purdue OWL, CliffsNotes, Sparknotes, Lecturia

## 🛠 Technologies

### Backend
- FastAPI (Python 3.10+)
- MongoDB (Motor async)
- Claude Sonnet 4 via Emergent Universal Key
- Gemini 2.0 Flash pour analyse de documents

### Frontend
- React 19
- Tailwind CSS + Shadcn/UI
- Axios

## 🚀 Installation Locale

### Prérequis
- Python 3.10+
- Node.js 18+
- MongoDB
- Emergent LLM Key

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Éditer .env avec vos configurations
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

### Frontend

```bash
cd frontend
yarn install
cp .env.example .env
# Éditer .env avec l'URL de votre backend
yarn start
```

## 🔑 Emergent LLM Key

Obtenez votre clé universelle sur https://emergentagent.com
- Profile → Universal Key
- Fonctionne avec OpenAI, Claude, Gemini

## 📝 Licence

MIT License - © 2025 Collège Champagneur

---

**Étienne - Votre compagnon d'études intelligent** 🎓
