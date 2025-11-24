import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Badge } from './components/ui/badge';
import { ScrollArea } from './components/ui/scroll-area';
import { Separator } from './components/ui/separator';
import { Avatar, AvatarFallback } from './components/ui/avatar';
import { toast } from 'sonner';
import { Toaster } from './components/ui/sonner';
import ConversationSidebar from './ConversationSidebar';
import ConversationHistory from './ConversationHistory';
import AuthModal from './components/AuthModal';
import AdminPanel from './AdminPanel';
import { formatMessage, cleanMessageForExport } from './utils/formatMessage';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [messages, setMessages] = useState([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [activeTab, setActiveTab] = useState('je_veux');
  const [isLoading, setIsLoading] = useState(false);
  const [subjects, setSubjects] = useState({});
  const [uploadedFile, setUploadedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [textToAnalyze, setTextToAnalyze] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const messagesEndRef = useRef(null);
  
  // États pour l'historique des conversations
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  
  // États pour l'authentification et l'admin
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [showAdminPanel, setShowAdminPanel] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);
  const [isAdmin, setIsAdmin] = useState(false);

  const messageTypes = {
    plans_cours: {
      title: 'Plans de cours',
      description: 'Créez des planifications détaillées adaptées au programme (Sec 1-5)',
      placeholder: 'Ex: Crée un plan de cours sur la photosynthèse pour Secondaire 4...',
      icon: '📚'
    },
    evaluations: {
      title: 'Évaluations',
      description: 'Générez examens, quiz, grilles de correction professionnelles',
      placeholder: 'Ex: Génère un examen de français Sec 3 sur le roman avec corrigé...',
      icon: '📝'
    },
    activites: {
      title: 'Activités',
      description: 'Créez exercices, projets, activités pédagogiques engageantes',
      placeholder: 'Ex: Propose 3 activités interactives sur les fractions pour Sec 1...',
      icon: '🎯'
    },
    ressources: {
      title: 'Ressources',
      description: 'Trouvez idées, matériel pédagogique, sources fiables',
      placeholder: 'Ex: Trouve-moi des idées de projets en histoire Sec 4...',
      icon: '🔍'
    },
    outils: {
      title: 'Outils',
      description: 'Différenciation, grilles d\'évaluation, planification, vérification IA',
      placeholder: 'Ex: Crée une grille d\'évaluation pour un exposé oral ou vérifie un texte pour plagiat...',
      icon: '⚙️'
    }
  };

  useEffect(() => {
    fetchSubjects();
    // Génération d'un nouvel ID de session
    setSessionId(Date.now().toString());
    
    // Configurer l'interceptor Axios pour ajouter le token JWT
    const requestInterceptor = axios.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('etienne_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Interceptor pour gérer les erreurs 401 (non authentifié)
    const responseInterceptor = axios.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401 || error.response?.status === 403) {
          // Token invalide ou expiré, déconnecter l'utilisateur
          localStorage.removeItem('etienne_token');
          localStorage.removeItem('etienne_user');
          setCurrentUser(null);
          setIsAdmin(false);
          setShowAuthModal(true);
          toast.error('Session expirée, veuillez vous reconnecter');
        }
        return Promise.reject(error);
      }
    );
    
    // Vérifier si l'utilisateur est connecté
    const storedUser = localStorage.getItem('etienne_user');
    const storedToken = localStorage.getItem('etienne_token');
    if (storedUser && storedToken) {
      const user = JSON.parse(storedUser);
      setCurrentUser(user);
      
      // Vérifier si c'est un admin
      const adminEmails = ['informatique@champagneur.qc.ca'];
      setIsAdmin(adminEmails.includes(user.email));
    }
    // Ne plus ouvrir automatiquement la modale de connexion

    // Cleanup interceptors on unmount
    return () => {
      axios.interceptors.request.eject(requestInterceptor);
      axios.interceptors.response.eject(responseInterceptor);
    };
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      // Utiliser scrollTop pour scroller uniquement dans le conteneur, pas la page
      const scrollArea = messagesEndRef.current.closest('[data-radix-scroll-area-viewport]');
      if (scrollArea) {
        scrollArea.scrollTop = scrollArea.scrollHeight;
      } else {
        messagesEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }
  };

  const fetchSubjects = async () => {
    try {
      const response = await axios.get(`${API}/subjects`);
      setSubjects(response.data);
    } catch (error) {
      console.error('Erreur lors du chargement des matières:', error);
    }
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!currentMessage.trim() || isLoading) return;

    const messageToSend = currentMessage;
    setCurrentMessage('');
    setIsLoading(true);

    // Créer une nouvelle conversation si c'est le premier message
    if (messages.length === 0 && !currentConversationId) {
      const convId = ConversationHistory.createConversation(messageToSend);
      setCurrentConversationId(convId);
    }

    // Ajouter le message utilisateur
    const userMessage = {
      id: Date.now(),
      message: messageToSend,
      isUser: true,
      type: activeTab,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    
    // Sauvegarder dans l'historique
    if (currentConversationId) {
      ConversationHistory.addMessage(currentConversationId, {
        role: 'user',
        content: messageToSend
      });
    }

    try {
      // Envoyer le message au backend (qui gère automatiquement les images)
      const response = await axios.post(`${API}/chat`, {
        message: messageToSend,
        message_type: activeTab,
        session_id: sessionId
      });

      // Ajouter la réponse IA
      const aiMessage = {
        id: response.data.id,
        message: response.data.response,
        isUser: false,
        type: activeTab,
        trust_score: response.data.trust_score,
        sources: response.data.sources,
        image_base64: response.data.image_base64,  // Image si présente
        can_download: response.data.can_download || response.data.response.length > 100,
        timestamp: new Date(response.data.timestamp)
      };
      setMessages(prev => [...prev, aiMessage]);
      
      // Sauvegarder dans l'historique
      if (currentConversationId) {
        ConversationHistory.addMessage(currentConversationId, {
          role: 'assistant',
          content: response.data.response
        });
      }

      if (response.data.trust_score) {
        toast.success(`Sources analysées - Fiabilité: ${Math.round(response.data.trust_score * 100)}%`);
      }
      
      if (response.data.image_base64) {
        toast.success('🎨 Image générée avec succès!');
      }

    } catch (error) {
      console.error('Erreur:', error);
      toast.error('Erreur lors de l\'envoi du message');
      
      const errorMessage = {
        id: Date.now() + 1,
        message: 'Désolé, une erreur s\'est produite. Veuillez réessayer.',
        isUser: false,
        type: activeTab,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Vérifier la taille du fichier (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      toast.error('Fichier trop volumineux. Taille maximale: 10MB');
      return;
    }

    // Vérifier le format
    const allowedExtensions = ['pdf', 'docx', 'doc', 'txt', 'xlsx', 'xls', 'csv', 'pptx'];
    const fileExtension = file.name.split('.').pop().toLowerCase();
    
    if (!allowedExtensions.includes(fileExtension)) {
      toast.error('Format non supporté. Formats acceptés: PDF, DOCX, TXT, XLSX, CSV, PPTX');
      return;
    }

    setIsUploading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(`${API}/upload-file`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setUploadedFile({
        name: response.data.filename,
        extracted_text: response.data.extracted_text,
        text_length: response.data.text_length
      });

      toast.success(`📎 Fichier "${response.data.filename}" analysé ! Posez votre question.`);
      
    } catch (error) {
      console.error('Erreur upload:', error);
      toast.error('Erreur lors de l\'analyse du fichier');
    } finally {
      setIsUploading(false);
      event.target.value = ''; // Reset input
    }
  };

  const sendMessageWithFile = async (e) => {
    e.preventDefault();
    if (!currentMessage.trim() || isLoading) return;

    if (uploadedFile) {
      // Envoyer message avec analyse de fichier
      setIsLoading(true);
      
      const userMessage = {
        id: Date.now(),
        message: `📎 ${uploadedFile.name}: ${currentMessage}`,
        isUser: true,
        type: activeTab,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, userMessage]);
      
      const messageToSend = currentMessage;
      setCurrentMessage('');

      try {
        const response = await axios.post(`${API}/analyze-file`, {
          question: messageToSend,
          extracted_text: uploadedFile.extracted_text,
          filename: uploadedFile.name,
          message_type: activeTab
        });

        const aiMessage = {
          id: response.data.id,
          message: response.data.response,
          isUser: false,
          type: activeTab,
          trust_score: response.data.trust_score,
          sources: response.data.sources,
          can_download: true,
          timestamp: new Date(response.data.timestamp)
        };
        setMessages(prev => [...prev, aiMessage]);

        // Réinitialiser le fichier uploadé après utilisation
        setUploadedFile(null);
        toast.success('Analyse du document terminée !');

      } catch (error) {
        console.error('Erreur analyse:', error);
        toast.error('Erreur lors de l\'analyse du fichier');
      } finally {
        setIsLoading(false);
      }
    } else {
      // Message normal sans fichier
      sendMessage(e);
    }
  };

  const getTrustBadge = (trustScore) => {
    if (!trustScore) return null;
    
    const percentage = Math.round(trustScore * 100);
    let variant = 'secondary';
    let text = '';
    
    if (percentage >= 80) {
      variant = 'default';
      text = `Très fiable (${percentage}%)`;
    } else if (percentage >= 60) {
      variant = 'secondary';
      text = `Fiable (${percentage}%)`;
    } else {
      variant = 'destructive';
      text = `Modérément fiable (${percentage}%)`;
    }
    
    return (
      <Badge variant={variant} className="text-xs">
        {text}
      </Badge>
    );
  };

  const downloadDocument = async (content, title, format) => {
    try {
      setIsLoading(true);
      
      // Nettoyer le contenu avant l'export (enlever les phrases d'intro)
      const cleanedContent = cleanMessageForExport(content);
      
      const response = await axios.post(`${API}/generate-document`, {
        content: cleanedContent,
        title: title || 'Document Étienne',
        format: format,
        filename: `etienne_document_${Date.now()}`
      }, {
        responseType: 'blob'
      });

      // Créer un lien de téléchargement
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      
      const extensions = {
        'pdf': 'pdf',
        'docx': 'docx', 
        'pptx': 'pptx',
        'xlsx': 'xlsx'
      };
      
      link.setAttribute('download', `etienne_document_${Date.now()}.${extensions[format]}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      toast.success(`Document ${format.toUpperCase()} téléchargé avec succès !`);
      
    } catch (error) {
      console.error('Erreur téléchargement:', error);
      toast.error('Erreur lors du téléchargement du document');
    } finally {
      setIsLoading(false);
    }
  };

  const analyzeText = async (analysisType = 'complete') => {
    if (!textToAnalyze.trim()) {
      toast.error('Veuillez entrer du texte à analyser');
      return;
    }

    setIsAnalyzing(true);
    
    try {
      let endpoint = '';
      switch (analysisType) {
        case 'ai':
          endpoint = '/detect-ai';
          break;
        case 'plagiarism':
          endpoint = '/check-plagiarism';
          break;
        default:
          endpoint = '/analyze-text';
      }

      const response = await axios.post(`${API}${endpoint}`, {
        text: textToAnalyze
      });

      setAnalysisResult(response.data);
      
      // Ajouter le résultat aux messages
      const analysisMessage = {
        id: Date.now(),
        message: `Analyse de texte (${textToAnalyze.substring(0, 100)}${textToAnalyze.length > 100 ? '...' : ''})`,
        isUser: true,
        type: 'verification',
        timestamp: new Date()
      };

      const resultMessage = {
        id: Date.now() + 1,
        message: formatAnalysisResult(response.data),
        isUser: false,
        type: 'verification',
        analysis_result: response.data,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, analysisMessage, resultMessage]);
      
      toast.success('Analyse terminée !');
      
    } catch (error) {
      console.error('Erreur analyse:', error);
      toast.error('Erreur lors de l\'analyse du texte');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const formatAnalysisResult = (result) => {
    if (result.overall_assessment) {
      // Analyse complète
      const { ai_detection, plagiarism_check, overall_assessment, language } = result;
      
      return `## 📊 Analyse complète du texte

**🌐 Langue détectée:** ${language === 'en' ? 'Anglais' : 'Français'}

**🤖 Détection IA:**
- Probabilité: ${Math.round(ai_detection.ai_probability * 100)}%
- Statut: ${ai_detection.is_likely_ai ? '⚠️ Probablement généré par IA' : '✅ Semble authentique'}
- Confiance: ${ai_detection.confidence}

**📝 Vérification de plagiat:**
- Risque: ${Math.round(plagiarism_check.plagiarism_risk * 100)}%
- Niveau: ${plagiarism_check.risk_level}
- Diversité vocabulaire: ${Math.round(plagiarism_check.vocabulary_diversity * 100)}%
- Statut: ${plagiarism_check.is_suspicious ? '⚠️ Suspect' : '✅ Semble original'}

**🎯 Évaluation globale:**
- Niveau de risque: ${overall_assessment.risk_level}
- Nombre de mots: ${overall_assessment.word_count}

**💡 Recommandations:**
${overall_assessment.recommendations.map(rec => `- ${rec}`).join('\n')}`;
    } else if (result.ai_detection) {
      // Détection IA seulement
      const { ai_detection } = result;
      return `## 🤖 Détection d'IA

**Probabilité IA:** ${Math.round(ai_detection.ai_probability * 100)}%
**Statut:** ${ai_detection.is_likely_ai ? '⚠️ Probablement généré par IA' : '✅ Semble authentique'}
**Confiance:** ${ai_detection.confidence}

${ai_detection.detected_patterns?.length > 0 ? `**Patterns détectés:** ${ai_detection.detected_patterns.join(', ')}` : ''}`;
    } else if (result.plagiarism_result) {
      // Vérification plagiat seulement
      const { plagiarism_result } = result;
      return `## 📝 Vérification de plagiat

**Risque de plagiat:** ${Math.round(plagiarism_result.plagiarism_risk * 100)}%
**Niveau:** ${plagiarism_result.risk_level}
**Diversité vocabulaire:** ${Math.round(plagiarism_result.vocabulary_diversity * 100)}%
**Statut:** ${plagiarism_result.is_suspicious ? '⚠️ Suspect' : '✅ Semble original'}

**Recommandation:** ${plagiarism_result.recommendation}`;
    }
    
    return 'Analyse terminée';
  };

  // Fonctions pour gérer l'historique des conversations
  const handleNewConversation = () => {
    setMessages([]);
    setCurrentConversationId(null);
    setSessionId(Date.now().toString());
    setUploadedFile(null);
    toast.success('Nouvelle conversation démarrée');
  };

  const handleSelectConversation = (convId) => {
    const conversations = ConversationHistory.getAllConversations();
    const conv = conversations[convId];
    
    if (!conv) return;
    
    // Charger les messages de la conversation
    const loadedMessages = conv.messages.map((msg, index) => ({
      id: Date.now() + index,
      message: msg.content,
      isUser: msg.role === 'user',
      type: activeTab,
      timestamp: new Date(msg.timestamp)
    }));
    
    setMessages(loadedMessages);
    setCurrentConversationId(convId);
    setIsSidebarOpen(false);
    toast.success(`Conversation "${conv.title}" chargée`);
  };

  const handleDeleteConversation = (convId) => {
    ConversationHistory.deleteConversation(convId);
    if (currentConversationId === convId) {
      handleNewConversation();
    }
  };

  // Fonctions d'authentification
  const handleAuthSuccess = (user) => {
    setCurrentUser(user);
    const adminEmails = ['informatique@champagneur.qc.ca'];
    setIsAdmin(adminEmails.includes(user.email));
    toast.success(`Bienvenue ${user.full_name} !`);
  };

  const handleLogout = () => {
    localStorage.removeItem('etienne_token');
    localStorage.removeItem('etienne_user');
    setCurrentUser(null);
    setIsAdmin(false);
    setShowAuthModal(true);
    toast.success('Déconnexion réussie');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-white to-blue-50">
      <Toaster position="top-right" />
      
      {/* Modal d'authentification */}
      <AuthModal 
        isOpen={showAuthModal}
        onClose={() => setShowAuthModal(false)}
        onSuccess={handleAuthSuccess}
      />
      
      {/* Panneau d'administration */}
      {showAdminPanel && (
        <AdminPanel 
          onClose={() => setShowAdminPanel(false)}
        />
      )}
      
      {/* Sidebar d'historique */}
      <ConversationSidebar
        isOpen={isSidebarOpen}
        onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
        currentConversationId={currentConversationId}
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
        onDeleteConversation={handleDeleteConversation}
      />
      
      {/* Header */}
      <header className="bg-white/90 backdrop-blur-md border-b border-orange-100 sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-orange-500 to-blue-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-lg">É</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Étienne</h1>
                <p className="text-sm text-gray-600">Assistant IA pour les enseignants du secondaire québécois (Sec 1 à 5)</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                ✅ Programme québécois
              </Badge>
              <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                👨‍🏫 Pour enseignants
              </Badge>
              
              {/* Boutons Auth/Admin */}
              {currentUser ? (
                <>
                  <Badge variant="outline" className="bg-purple-50 text-purple-700 border-purple-200">
                    👤 {currentUser.full_name}
                  </Badge>
                  {isAdmin && (
                    <Button 
                      size="sm" 
                      onClick={() => setShowAdminPanel(true)}
                      className="bg-orange-600 hover:bg-orange-700"
                    >
                      👨‍💼 Admin
                    </Button>
                  )}
                  <Button 
                    size="sm" 
                    variant="outline"
                    onClick={handleLogout}
                  >
                    🚪 Déconnexion
                  </Button>
                </>
              ) : (
                <Button 
                  size="sm"
                  onClick={() => setShowAuthModal(true)}
                >
                  🔐 Connexion
                </Button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative py-12 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <div className="mb-8">
            <img 
              src="https://images.unsplash.com/photo-1614492898637-435e0f87cef8?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NDk1ODF8MHwxfHNlYXJjaHwyfHwlQzMlQTl0dWRpYW50JTIwbHljJUMzJUE5ZXxlbnwwfHx8fDE3NTk0MTA1OTF8MA&ixlib=rb-4.1.0&q=85" 
              alt="Étudiant avec technologie" 
              className="w-32 h-32 rounded-full mx-auto object-cover shadow-lg"
            />
          </div>
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Étienne, votre assistant IA pour <span className="text-transparent bg-clip-text bg-gradient-to-r from-orange-500 to-blue-600">l'enseignement au secondaire</span>
          </h2>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto leading-relaxed">
            Créez des plans de cours, générez des évaluations professionnelles, trouvez des idées de projets et accédez à des ressources pédagogiques adaptées au programme du secondaire québécois (Sec 1 à 5).
          </p>
        </div>
      </section>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-4 pb-8">
        {/* Vérification de la connexion */}
        {!currentUser ? (
          <Card className="max-w-2xl mx-auto bg-white/90 backdrop-blur-sm border-orange-200">
            <CardContent className="p-12 text-center">
              <div className="text-6xl mb-4">🔐</div>
              <h3 className="text-2xl font-bold text-gray-900 mb-3">
                Connexion Requise
              </h3>
              <p className="text-gray-600 mb-6">
                Vous devez vous connecter avec une licence valide pour accéder à Étienne.
              </p>
              <Button 
                size="lg"
                onClick={() => setShowAuthModal(true)}
                className="bg-gradient-to-r from-orange-500 to-blue-600 hover:from-orange-600 hover:to-blue-700"
              >
                🔐 Se connecter
              </Button>
            </CardContent>
          </Card>
        ) : (
          <div className="grid lg:grid-cols-4 gap-6">
            
            {/* Sidebar - Matières */}
            <div className="lg:col-span-1">
            <Card className="bg-white/80 backdrop-blur-sm border-orange-100">
              <CardHeader>
                <CardTitle className="text-lg text-gray-900">📚 Matières scolaires</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {Object.entries(subjects).map(([key, category]) => (
                    <div key={key} className="space-y-2">
                      <h4 className="font-semibold text-sm text-gray-700">{category.name}</h4>
                      <div className="flex flex-wrap gap-1">
                        {category.subjects?.map((subject) => (
                          <Badge key={subject} variant="outline" className="text-xs bg-gray-50 hover:bg-gray-100 cursor-pointer transition-colors">
                            {subject}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Chat Interface */}
          <div className="lg:col-span-3">
            <Card className="bg-white/90 backdrop-blur-sm border-orange-100 min-h-[600px] flex flex-col">
              <CardHeader>
                <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
                  <TabsList className="grid w-full grid-cols-5 bg-gray-50">
                    {Object.entries(messageTypes).map(([key, type]) => (
                      <TabsTrigger key={key} value={key} className="text-sm flex items-center gap-1 data-[state=active]:bg-white data-[state=active]:text-gray-900">
                        <span className="text-xs">{type.icon}</span>
                        <span className="hidden sm:inline">{type.title}</span>
                      </TabsTrigger>
                    ))}
                  </TabsList>
                  
                  {Object.entries(messageTypes).map(([key, type]) => (
                    <TabsContent key={key} value={key} className="mt-4">
                      <div className="text-center p-4 bg-gradient-to-r from-orange-50 to-blue-50 rounded-lg">
                        <h3 className="font-semibold text-gray-900 mb-2">{type.icon} {type.title}</h3>
                        <p className="text-sm text-gray-600">{type.description}</p>
                      </div>
                    </TabsContent>
                  ))}
                </Tabs>
              </CardHeader>
              
              <CardContent className="flex-1 flex flex-col">
                {/* Messages */}
                <ScrollArea 
                  className="flex-1 mb-4 border-2 border-orange-200 rounded-lg" 
                  style={{ 
                    maxHeight: '400px',
                    overflow: 'auto',
                    scrollbarWidth: 'auto',
                    scrollbarColor: '#f97316 #fef3e2'
                  }}
                >
                  <div 
                    className="space-y-4 pr-2" 
                    style={{
                      height: '400px',
                      overflow: 'auto',
                      paddingRight: '8px',
                      scrollbarWidth: 'auto',
                      scrollbarColor: '#f97316 #fef3e2'
                    }}
                  >
                    {messages.length === 0 ? (
                      <div className="text-center py-8">
                        <div className="mb-4">
                          <img 
                            src="https://images.unsplash.com/photo-1757143137415-0790a01bfa6d?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NDk1ODF8MHwxfHNlYXJjaHwzfHwlQzMlQTl0dWRpYW50JTIwbHljJUMzJUE5ZXxlbnwwfHx8fDE3NTk0MTA1OTF8MA&ixlib=rb-4.1.0&q=85" 
                            alt="Étudiante souriante" 
                            className="w-20 h-20 rounded-full mx-auto object-cover"
                          />
                        </div>
                        <p className="text-gray-500 mb-4">Commencez à créer du matériel pédagogique</p>
                        <p className="text-sm text-gray-400">Choisissez un type d'outil ci-dessus pour générer plans de cours, évaluations, activités...</p>
                      </div>
                    ) : (
                      messages.map((msg) => (
                        <div key={msg.id} className={`flex ${msg.isUser ? 'justify-end' : 'justify-start'}`}>
                          <div className={`max-w-[80%] ${msg.isUser ? 'order-2' : 'order-1'}`}>
                            <div className={`flex items-start gap-3 ${msg.isUser ? 'flex-row-reverse' : 'flex-row'}`}>
                              <Avatar className="w-8 h-8">
                                <AvatarFallback className={msg.isUser ? 'bg-blue-500 text-white' : 'bg-orange-500 text-white'}>
                                  {msg.isUser ? '👤' : '🤖'}
                                </AvatarFallback>
                              </Avatar>
                              <div className={`rounded-2xl px-4 py-3 ${msg.isUser 
                                ? 'bg-blue-500 text-white' 
                                : 'bg-gray-100 text-gray-900'
                              }`}>
                                {msg.isUser ? (
                                  <p className="text-sm leading-relaxed">{msg.message}</p>
                                ) : (
                                  <div 
                                    className="text-sm leading-relaxed formatted-message" 
                                    dangerouslySetInnerHTML={{__html: formatMessage(msg.message)}}
                                  />
                                )}
                                
                                {/* Affichage de l'image générée */}
                                {msg.image_base64 && (
                                  <div className="mt-3">
                                    <img 
                                      src={`data:image/png;base64,${msg.image_base64}`} 
                                      alt="Image générée par IA" 
                                      className="rounded-lg max-w-full h-auto"
                                    />
                                    <a 
                                      href={`data:image/png;base64,${msg.image_base64}`}
                                      download={`etienne_image_${Date.now()}.png`}
                                      className="inline-block mt-2 text-xs bg-green-100 hover:bg-green-200 text-green-700 px-3 py-1 rounded transition-colors"
                                    >
                                      💾 Télécharger l'image
                                    </a>
                                  </div>
                                )}
                                
                                {msg.trust_score && (
                                  <div className="mt-2">
                                    {getTrustBadge(msg.trust_score)}
                                  </div>
                                )}
                                {!msg.isUser && msg.can_download && (
                                  <div className="mt-3 pt-2 border-t border-gray-200">
                                    <p className="text-xs text-gray-600 mb-2">📥 Télécharger cette réponse :</p>
                                    <div className="flex gap-1 flex-wrap">
                                      <button
                                        onClick={() => downloadDocument(msg.message, 'Réponse Étienne', 'pdf')}
                                        className="text-xs bg-red-100 hover:bg-red-200 text-red-700 px-2 py-1 rounded transition-colors"
                                        disabled={isLoading}
                                      >
                                        📄 PDF
                                      </button>
                                      <button
                                        onClick={() => downloadDocument(msg.message, 'Réponse Étienne', 'docx')}
                                        className="text-xs bg-blue-100 hover:bg-blue-200 text-blue-700 px-2 py-1 rounded transition-colors"
                                        disabled={isLoading}
                                      >
                                        📝 Word
                                      </button>
                                      <button
                                        onClick={() => downloadDocument(msg.message, 'Réponse Étienne', 'pptx')}
                                        className="text-xs bg-orange-100 hover:bg-orange-200 text-orange-700 px-2 py-1 rounded transition-colors"
                                        disabled={isLoading}
                                      >
                                        📊 PowerPoint
                                      </button>
                                      <button
                                        onClick={() => downloadDocument(msg.message, 'Réponse Étienne', 'xlsx')}
                                        className="text-xs bg-green-100 hover:bg-green-200 text-green-700 px-2 py-1 rounded transition-colors"
                                        disabled={isLoading}
                                      >
                                        📈 Excel
                                      </button>
                                    </div>
                                  </div>
                                )}
                              </div>
                            </div>
                            <div className={`text-xs text-gray-400 mt-1 ${msg.isUser ? 'text-right' : 'text-left'}`}>
                              {new Date(msg.timestamp).toLocaleTimeString('fr-FR', { 
                                hour: '2-digit', 
                                minute: '2-digit' 
                              })}
                            </div>
                          </div>
                        </div>
                      ))
                    )}
                    {isLoading && (
                      <div className="flex justify-start">
                        <div className="flex items-start gap-3">
                          <Avatar className="w-8 h-8">
                            <AvatarFallback className="bg-orange-500 text-white">🤖</AvatarFallback>
                          </Avatar>
                          <div className="bg-gray-100 rounded-2xl px-4 py-3">
                            <div className="flex space-x-1">
                              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                    <div ref={messagesEndRef} />
                  </div>
                </ScrollArea>
                
                <Separator className="mb-4" />
                
                {/* Zone d'upload de fichier */}
                {uploadedFile && (
                  <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-blue-600">📎</span>
                        <div>
                          <p className="text-sm font-medium text-blue-800">{uploadedFile.name}</p>
                          <p className="text-xs text-blue-600">{uploadedFile.text_length} caractères extraits</p>
                        </div>
                      </div>
                      <button
                        onClick={() => setUploadedFile(null)}
                        className="text-blue-600 hover:text-blue-800 p-1"
                        title="Supprimer le fichier"
                      >
                        ×
                      </button>
                    </div>
                  </div>
                )}
                
                {/* Interface spéciale pour la vérification de texte */}
                {activeTab === 'verification' ? (
                  <div className="space-y-4">
                    <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                      <h4 className="font-semibold text-yellow-800 mb-2">🔍 Vérification de texte</h4>
                      <p className="text-sm text-yellow-700">Collez votre texte ci-dessous pour détecter s'il a été généré par IA et vérifier l'originalité.</p>
                    </div>
                    
                    <textarea
                      value={textToAnalyze}
                      onChange={(e) => setTextToAnalyze(e.target.value)}
                      placeholder="Collez votre texte ici pour l'analyser..."
                      className="w-full h-32 p-3 border border-gray-200 rounded-lg resize-none focus:border-orange-300 focus:ring-orange-200"
                      disabled={isAnalyzing}
                    />
                    
                    <div className="flex gap-2">
                      <Button 
                        onClick={() => analyzeText('complete')}
                        disabled={!textToAnalyze.trim() || isAnalyzing}
                        className="bg-gradient-to-r from-orange-500 to-blue-600 hover:from-orange-600 hover:to-blue-700 text-white font-medium px-4 transition-all duration-200"
                      >
                        {isAnalyzing ? '🔄 Analyse...' : '🔍 Analyse complète'}
                      </Button>
                      
                      <Button 
                        onClick={() => analyzeText('ai')}
                        disabled={!textToAnalyze.trim() || isAnalyzing}
                        variant="outline"
                        className="border-orange-300 text-orange-600 hover:bg-orange-50"
                      >
                        🤖 Détection IA
                      </Button>
                      
                      <Button 
                        onClick={() => analyzeText('plagiarism')}
                        disabled={!textToAnalyze.trim() || isAnalyzing}
                        variant="outline"
                        className="border-blue-300 text-blue-600 hover:bg-blue-50"
                      >
                        📝 Plagiat
                      </Button>
                    </div>
                    
                    <div className="text-xs text-gray-500">
                      💡 L'analyse détecte les patterns d'IA et vérifie l'originalité du contenu
                    </div>
                  </div>
                ) : (
                  /* Interface normale pour les autres onglets */
                  <form onSubmit={sendMessageWithFile} className="space-y-2">
                    <div className="flex gap-2">
                      <Input
                        data-testid="chat-input"
                        value={currentMessage}
                        onChange={(e) => setCurrentMessage(e.target.value)}
                        placeholder={
                          uploadedFile 
                            ? `Posez votre question à Étienne sur "${uploadedFile.name}"...`
                            : messageTypes[activeTab]?.placeholder || "Parlez à Étienne..."
                        }
                        disabled={isLoading || isUploading}
                        className="flex-1 bg-white border-gray-200 focus:border-orange-300 focus:ring-orange-200"
                      />
                      
                      {/* Bouton upload */}
                      <label className="relative cursor-pointer" title="Joindre un fichier">
                        <input
                          type="file"
                          onChange={handleFileUpload}
                          accept=".pdf,.docx,.doc,.txt,.xlsx,.xls,.csv,.pptx"
                          className="hidden"
                          disabled={isUploading}
                        />
                        <div className={`
                          flex items-center justify-center w-12 h-10 rounded-lg border-2 border-dashed transition-all
                          ${isUploading 
                            ? 'border-gray-300 bg-gray-100 cursor-not-allowed' 
                            : 'border-[#FF8C42] bg-[#FFE5D9] hover:bg-[#FFD4C0] hover:border-[#FF7A29]'
                          }
                        `}>
                          {isUploading ? (
                            <div className="w-4 h-4 border-2 border-gray-400 border-t-transparent rounded-full animate-spin"></div>
                          ) : (
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="text-gray-600">
                              <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                            </svg>
                          )}
                        </div>
                      </label>
                      
                      <Button 
                        data-testid="send-button"
                        type="submit" 
                        disabled={!currentMessage.trim() || isLoading || isUploading}
                        className="bg-gradient-to-r from-orange-500 to-blue-600 hover:from-orange-600 hover:to-blue-700 text-white font-medium px-6 transition-all duration-200"
                      >
                        {isLoading ? '...' : uploadedFile ? 'Analyser' : 'Envoyer'}
                      </Button>
                    </div>
                    
                    {/* Info formats supportés */}
                    <div className="text-xs text-gray-500 flex items-center gap-2">
                      <span>📎 Formats supportés: PDF, Word, Excel, PowerPoint, TXT, CSV</span>
                      <span>•</span>
                      <span>Max: 10MB</span>
                    </div>
                  </form>
                )}
              </CardContent>
            </Card>
          </div>
          
          {/* Features Section - full width in the grid */}
          <div className="lg:col-span-4 mt-8">
            <h3 className="text-2xl font-bold text-center text-gray-900 mb-8">Fonctionnalités principales</h3>
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {Object.entries(messageTypes).map(([key, type]) => (
              <Card key={key} className="bg-white/80 backdrop-blur-sm border-orange-100 hover:shadow-lg transition-shadow cursor-pointer" 
                    onClick={() => setActiveTab(key)}>
                <CardContent className="p-6 text-center">
                  <div className="text-3xl mb-3">{type.icon}</div>
                  <h4 className="font-semibold text-gray-900 mb-2">{type.title}</h4>
                  <p className="text-sm text-gray-600">{type.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
        
        {/* Educational Image */}
        {/* Section Fonctionnalités */}
        <div className="mt-12 space-y-8">
          
          {/* Upload de fichiers */}
          <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-xl p-6 border border-green-200">
            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              📤 Nouveau : Analysez vos documents
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold text-gray-800 mb-2">Comment utiliser :</h4>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• Cliquez sur l'icône 📎 à côté du champ de message</li>
                  <li>• Sélectionnez votre document (PDF, Word, Excel, etc.)</li>
                  <li>• Posez votre question sur le contenu</li>
                  <li>• Étienne analyse et répond en se basant sur votre fichier</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-gray-800 mb-2">Exemples pour enseignants :</h4>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• "Résume ce manuel scolaire pour créer un plan de cours"</li>
                  <li>• "Extrais les concepts clés de ce chapitre"</li>
                  <li>• "Crée des questions d'examen basées sur ce document"</li>
                  <li>• "Génère des exercices à partir de ce contenu"</li>
                </ul>
              </div>
            </div>
            <div className="mt-4 p-3 bg-white/60 rounded-lg">
              <p className="text-sm text-gray-700">
                <strong>Formats supportés :</strong> PDF, Word (.docx), Excel (.xlsx), PowerPoint (.pptx), 
                Texte (.txt), CSV • <strong>Taille max :</strong> 10MB
              </p>
            </div>
          </div>

          {/* Téléchargement de documents */}
          <div className="bg-gradient-to-r from-blue-50 to-orange-50 rounded-xl p-6 border border-orange-200">
            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              📥 Téléchargement de documents
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold text-gray-800 mb-2">Comment ça marche :</h4>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• Posez votre question à Étienne</li>
                  <li>• Des boutons de téléchargement apparaîtront sous les réponses</li>
                  <li>• Choisissez le format : PDF, Word, PowerPoint ou Excel</li>
                  <li>• Le document se télécharge automatiquement</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-gray-800 mb-2">Exemples pour enseignants :</h4>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• "Génère un plan de cours sur la Révolution tranquille"</li>
                  <li>• "Crée un examen de mathématiques Sec 3 avec corrigé"</li>
                  <li>• "Fais une grille d'évaluation pour un exposé oral"</li>
                  <li>• "Prépare une présentation PowerPoint sur la photosynthèse"</li>
                </ul>
              </div>
            </div>
            </div>
          </div>
          
          {/* Image Section - full width in the grid */}
          <div className="lg:col-span-4 mt-8 text-center">
            <img 
              src="https://images.unsplash.com/photo-1596574027151-2ce81d85af3e?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NTY2NzB8MHwxfHNlYXJjaHw0fHxlZHVjYXRpb24lMjBsZWFybmluZ3xlbnwwfHx8fDE3NTk0MTA1OTh8MA&ixlib=rb-4.1.0&q=85" 
              alt="Environnement d'apprentissage" 
              className="w-full max-w-2xl mx-auto rounded-xl shadow-lg object-cover h-64"
            />
            <p className="text-gray-600 mt-4 italic">Enseignement moderne et collaboratif au secondaire québécois</p>
          </div>
        </div>
        )}
      </div>
    </div>
  );
}

export default App;
