# Étienne - Assistant IA pour les étudiants québécois

Étienne est un assistant d'intelligence artificielle conçu spécialement pour les étudiants du Québec, fourni par le Collège Champagneur.

## 🚀 Fonctionnalités

- **Chat IA Intelligent** : Réponses personnalisées et contextuelles
- **Sources Anglaises Internationales** : CliffsNotes, SparkNotes, Lecturia, Oxford, Cambridge
- **Détecteur d'IA** : Analyse si un texte est généré par IA
- **Vérificateur de Plagiat** : Évalue l'originalité des textes
- **Réponses Multilingues** : FR pour sources québécoises, EN pour sources internationales
- **Interface Moderne** : Design responsive et intuitive

## 🌍 Sources Spécialisées

### Anglais - Sources Internationales
- **Littérature :** CliffsNotes, SparkNotes, Lecturia, Project Gutenberg
- **Grammaire :** Oxford English Grammar, Cambridge Grammar, Purdue OWL
- **Académique :** Harvard Writing Center, MIT Writing Center, McGill

### Français - Sources Québécoises
- Sites gouvernementaux (.gouv.qc.ca)
- Ministère de l'Éducation (MEES)
- Universités québécoises
- BANQ (Bibliothèque nationale)

## 🛠 Technologies

- **Frontend :** HTML/CSS/JavaScript (statique)
- **Backend :** Python (Vercel Serverless)
- **Base de données :** MongoDB Atlas (optionnel)
- **IA :** Hugging Face (gratuit)

## 🚀 Déploiement Vercel

### Structure des fichiers :
```
etienne-assistant/
├── index.html          # Frontend
├── api/
│   └── index.py        # Backend API
├── requirements.txt    # Dépendances Python
└── README.md           # Documentation
```

### Variables d'environnement :
- `MONGO_URL` : URL MongoDB Atlas (optionnel)
- `HUGGINGFACE_TOKEN` : Token gratuit Hugging Face (optionnel)
- `DB_NAME` : etienne_free

## 📖 Utilisation

1. **Chat Standard :** Questions en français → Sources québécoises
2. **Chat Anglais :** Questions en anglais → Sources internationales
3. **Vérification :** Détection IA + analyse plagiat
4. **Sources Fiables :** Recommandations par matière

## 🎯 Exemples

### Questions Françaises :
```
"Explique-moi les mathématiques du secondaire"
→ Sources québécoises (MEES, universités)
```

### Questions Anglaises :
```
"Help me with Shakespeare literature"
→ CliffsNotes, SparkNotes, Lecturia

"I need English grammar help"
→ Oxford, Cambridge, Purdue OWL
```

### Vérification de Texte :
```
Coller un texte → Analyse IA + Plagiat + Langue
```

## 💰 Coût : 0€/mois

- Vercel : Gratuit
- MongoDB Atlas : Gratuit (512MB)
- Hugging Face : Gratuit

**Étienne - Votre assistant IA éducatif !** 🎓

© 2024 Collège Champagneur
