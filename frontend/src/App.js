import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Badge } from './components/ui/badge';
import { Separator } from './components/ui/separator';
import { toast } from 'sonner';
import { Toaster } from './components/ui/sonner';
import ConversationSidebar from './ConversationSidebar';
import ConversationService from './ConversationService';
import AuthModal from './components/AuthModal';
import AdminPanel from './AdminPanel';
import LicenseAdminPanel from './LicenseAdminPanel';
import QuotaDisplay from './QuotaDisplay';
import MathSymbolsDemo from './MathSymbolsDemo';
import TextCorrectionModal from './TextCorrectionModal';
import Nouveautes from './Nouveautes';
import ProfileModal from './ProfileModal';
import { cleanMessageForExport, cleanLatexForPrint } from './utils/formatMessage';
import { ChatMessage, LoadingIndicator } from './components/ChatMessage';
import { FeatureSection } from './components/FeatureSection';
import { HeaderNav } from './components/HeaderNav';
import 'katex/dist/katex.min.css';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [messages, setMessages] = useState([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [activeTab, setActiveTab] = useState('plans_cours');
  const [isLoading, setIsLoading] = useState(false);
  const [subjects, setSubjects] = useState({});
  const [uploadedFile, setUploadedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [textToAnalyze, setTextToAnalyze] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const messagesEndRef = useRef(null);
  
  // États pour l'historique des conversations (cloud)
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [conversationHistory, setConversationHistory] = useState([]); // Pour la mémoire
  
  // États pour l'authentification et l'admin
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [showAdminPanel, setShowAdminPanel] = useState(false);
  const [showLicenseAdminPanel, setShowLicenseAdminPanel] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);
  const [isAdmin, setIsAdmin] = useState(false);
  const [isLicenseAdmin, setIsLicenseAdmin] = useState(false);
  const [showMathSymbols, setShowMathSymbols] = useState(false);
  const [isChatFullscreen, setIsChatFullscreen] = useState(false);
  const [showProfileModal, setShowProfileModal] = useState(false);
  const [showCorrectionModal, setShowCorrectionModal] = useState(false);
  const [showNouveautes, setShowNouveautes] = useState(false);
  const [serverReady, setServerReady] = useState(false);
  const [serverWaking, setServerWaking] = useState(false);

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
      description: 'Plans d\'intervention, rapports, formulaires, procédures pour tout le personnel',
      placeholder: 'Ex: Crée une grille d\'évaluation pour un exposé oral, un plan d\'intervention TDAH, une lettre aux parents...',
      icon: '⚙️'
    }
  };

  useEffect(() => {
    // Réveil du serveur avec retry
    const wakeServer = async () => {
      setServerWaking(true);
      for (let attempt = 0; attempt < 10; attempt++) {
        try {
          await axios.get(`${API}/quota-status`, { timeout: 8000 });
          setServerReady(true);
          setServerWaking(false);
          return;
        } catch (err) {
          console.log(`Réveil du serveur... tentative ${attempt + 1}/10`);
          if (attempt < 9) await new Promise(r => setTimeout(r, 3000));
        }
      }
      setServerWaking(false);
      setServerReady(true); // Allow usage even if health check fails
    };
    wakeServer();

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
        // Ne pas traiter les erreurs 403 de license-admin comme des erreurs d'auth
        const isLicenseAdminEndpoint = error.config?.url?.includes('license-admin');
        
        if (error.response?.status === 401) {
          // Token invalide ou expiré, déconnecter l'utilisateur
          localStorage.removeItem('etienne_token');
          localStorage.removeItem('etienne_user');
          setCurrentUser(null);
          setIsAdmin(false);
          setIsLicenseAdmin(false);
          setShowAuthModal(true);
          toast.error('Session expirée, veuillez vous reconnecter');
        } else if (error.response?.status === 403 && !isLicenseAdminEndpoint) {
          // 403 sur d'autres endpoints = problème d'autorisation
          toast.error('Accès non autorisé');
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
      
      // Vérifier si c'est un admin de licence
      checkLicenseAdmin(storedToken);
    }
    // Ne plus ouvrir automatiquement la modale de connexion

    // Cleanup interceptors on unmount
    return () => {
      axios.interceptors.request.eject(requestInterceptor);
      axios.interceptors.response.eject(responseInterceptor);
    };
  }, []);

  // Fonction pour vérifier si l'utilisateur est admin de licence
  const checkLicenseAdmin = async (token) => {
    try {
      await axios.get(`${API}/license-admin/my-license`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      setIsLicenseAdmin(true);
    } catch (error) {
      // L'utilisateur n'est pas admin de licence, c'est normal
      setIsLicenseAdmin(false);
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      // Utiliser scrollTop pour scroller uniquement dans le conteneur, pas la page
      const scrollContainer = messagesEndRef.current.closest('.chat-scroll-container');
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      } else {
        messagesEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }
  };

  const handlePrintChat = () => {
    if (messages.length === 0) {
      toast.error('Aucun message à imprimer');
      return;
    }
    
    const printWindow = window.open('', '_blank');
    const currentDate = new Date().toLocaleDateString('fr-CA', {
      year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit'
    });
    
    const chatContent = messages.map(msg => {
      const sender = msg.isUser ? 'Vous' : 'Étienne';
      const bgColor = msg.isUser ? '#3b82f6' : '#f3f4f6';
      const textColor = msg.isUser ? 'white' : '#111827';
      // Nettoyer le LaTeX pour l'impression
      const cleanedMessage = cleanLatexForPrint(msg.message);
      return `
        <div style="margin-bottom: 16px; display: flex; justify-content: ${msg.isUser ? 'flex-end' : 'flex-start'};">
          <div style="max-width: 80%; background: ${bgColor}; color: ${textColor}; padding: 12px 16px; border-radius: 16px;">
            <div style="font-weight: bold; margin-bottom: 4px; font-size: 12px;">${sender}</div>
            <div style="white-space: pre-wrap; line-height: 1.5;">${cleanedMessage.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</div>
          </div>
        </div>
      `;
    }).join('');
    
    printWindow.document.write(`
      <!DOCTYPE html>
      <html>
      <head>
        <title>Conversation Étienne - ${currentDate}</title>
        <style>
          body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 20px; max-width: 800px; margin: 0 auto; }
          .header { text-align: center; border-bottom: 2px solid #f97316; padding-bottom: 20px; margin-bottom: 30px; }
          .header h1 { color: #f97316; margin: 0; }
          .header p { color: #666; margin-top: 8px; }
          .logo { font-size: 48px; margin-bottom: 10px; }
          @media print { body { padding: 0; } }
        </style>
      </head>
      <body>
        <div class="header">
          <div class="logo">🤖</div>
          <h1>Conversation avec Étienne</h1>
          <p>Imprimé le ${currentDate}</p>
        </div>
        ${chatContent}
      </body>
      </html>
    `);
    
    printWindow.document.close();
    printWindow.onload = () => {
      printWindow.print();
    };
  };

  const fetchSubjects = async () => {
    try {
      const response = await axios.get(`${API}/subjects`);
      setSubjects(response.data);
    } catch (error) {
      console.error('Erreur lors du chargement des matières:', error);
    }
  };

  const handleCorrectionSubmit = async (formattedMessage) => {
    setActiveTab('evaluations');
    setShowCorrectionModal(false);

    // Directly trigger the send flow with the formatted message
    if (!formattedMessage.trim() || isLoading) return;

    setCurrentMessage('');
    setIsLoading(true);

    const userMessage = {
      id: Date.now(),
      message: formattedMessage,
      isUser: true,
      type: 'evaluations',
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);

    const updatedHistory = [...conversationHistory, { role: 'user', content: formattedMessage }];
    setConversationHistory(updatedHistory);

    let convId = currentConversationId;
    if (currentUser) {
      try {
        if (!convId) {
          convId = await ConversationService.createConversation(formattedMessage);
          if (convId) setCurrentConversationId(convId);
        }
        if (convId) {
          ConversationService.addMessage(convId, {
            role: 'user',
            content: formattedMessage
          }).catch(err => console.warn('Erreur sauvegarde message:', err));
        }
      } catch (err) {
        console.warn('Erreur création conversation cloud:', err);
      }
    }

    try {
      const response = await axios.post(`${API}/chat`, {
        message: formattedMessage,
        message_type: 'evaluations',
        session_id: sessionId,
        conversation_id: convId,
        conversation_history: updatedHistory.slice(-10)
      });

      const aiMessage = {
        id: response.data.id,
        message: response.data.response,
        isUser: false,
        type: 'evaluations',
        trust_score: response.data.trust_score,
        sources: response.data.sources,
        image_base64: response.data.image_base64,
        images: response.data.images || [],
        can_download: response.data.can_download || response.data.response.length > 100,
        timestamp: new Date(response.data.timestamp)
      };
      setMessages(prev => [...prev, aiMessage]);
      setConversationHistory(prev => [...prev, { role: 'assistant', content: response.data.response }]);

      if (convId && currentUser) {
        ConversationService.addMessage(convId, {
          role: 'assistant',
          content: response.data.response,
          image_base64: response.data.image_base64
        }).catch(err => console.warn('Erreur sauvegarde réponse:', err));
      }
    } catch (error) {
      console.error('Erreur correction:', error);
      const errMsg = {
        id: Date.now() + 1,
        message: 'Désolé, une erreur s\'est produite lors de la correction. Veuillez réessayer.',
        isUser: false,
        type: 'evaluations',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!currentMessage.trim() || isLoading) return;

    const messageToSend = currentMessage;
    setCurrentMessage('');
    setIsLoading(true);

    // Ajouter le message utilisateur immédiatement
    const userMessage = {
      id: Date.now(),
      message: messageToSend,
      isUser: true,
      type: activeTab,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    
    // Mettre à jour l'historique local pour la mémoire
    const updatedHistory = [...conversationHistory, { role: 'user', content: messageToSend }];
    setConversationHistory(updatedHistory);

    // Créer/sauvegarder conversation en arrière-plan (non-bloquant)
    let convId = currentConversationId;
    if (currentUser) {
      try {
        if (!convId) {
          convId = await ConversationService.createConversation(messageToSend);
          if (convId) setCurrentConversationId(convId);
        }
        if (convId) {
          // Sauvegarder en arrière-plan sans bloquer
          ConversationService.addMessage(convId, {
            role: 'user',
            content: messageToSend
          }).catch(err => console.warn('Erreur sauvegarde message:', err));
        }
      } catch (err) {
        console.warn('Erreur création conversation cloud:', err);
      }
    }

    try {
      // Envoyer le message au backend avec retry pour cold start
      let response;
      let retries = 0;
      const maxRetries = 2;
      while (retries <= maxRetries) {
        try {
          response = await axios.post(`${API}/chat`, {
            message: messageToSend,
            message_type: activeTab,
            session_id: sessionId,
            conversation_id: convId,
            conversation_history: updatedHistory.slice(-10)
          }, { timeout: 120000 });
          break;
        } catch (retryErr) {
          if (retries < maxRetries && (!retryErr.response || retryErr.code === 'ECONNABORTED')) {
            retries++;
            toast.info(`Le serveur se réveille... tentative ${retries + 1}/${maxRetries + 1}`);
            await new Promise(r => setTimeout(r, 3000));
          } else {
            throw retryErr;
          }
        }
      }

      // Ajouter la réponse IA
      const aiMessage = {
        id: response.data.id,
        message: response.data.response,
        isUser: false,
        type: activeTab,
        trust_score: response.data.trust_score,
        sources: response.data.sources,
        image_base64: response.data.image_base64,
        images: response.data.images || [],  // Liste d'images multiples
        can_download: response.data.can_download || response.data.response.length > 100,
        timestamp: new Date(response.data.timestamp)
      };
      setMessages(prev => [...prev, aiMessage]);
      
      // Mettre à jour l'historique local pour la mémoire
      setConversationHistory(prev => [...prev, { role: 'assistant', content: response.data.response }]);
      
      // Sauvegarder la réponse en arrière-plan (non-bloquant)
      if (convId && currentUser) {
        ConversationService.addMessage(convId, {
          role: 'assistant',
          content: response.data.response,
          image_base64: response.data.image_base64
        }).catch(err => console.warn('Erreur sauvegarde réponse:', err));
      }

      if (response.data.trust_score) {
        toast.success(`Sources analysées - Fiabilité: ${Math.round(response.data.trust_score * 100)}%`);
      }
      
      if (response.data.image_base64 || (response.data.images && response.data.images.length > 0)) {
        toast.success('🎨 Image(s) générée(s) avec succès!');
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
    const newFiles = Array.from(event.target.files);
    if (newFiles.length === 0) return;

    const allowedExtensions = ['pdf', 'docx', 'doc', 'txt', 'xlsx', 'xls', 'csv', 'pptx'];
    const maxSize = 10 * 1024 * 1024; // 10MB
    const maxFiles = 5; // Maximum 5 fichiers

    // Récupérer les fichiers déjà uploadés (si existants)
    const existingFiles = uploadedFile?.names || [];
    const existingText = uploadedFile?.extracted_text || '';

    // Vérifier le nombre total de fichiers (existants + nouveaux)
    const totalFiles = existingFiles.length + newFiles.length;
    if (totalFiles > maxFiles) {
      toast.error(`Maximum ${maxFiles} fichiers au total. Vous avez déjà ${existingFiles.length} fichier(s).`);
      return;
    }

    // Vérifier chaque nouveau fichier
    for (const file of newFiles) {
      // Vérifier si le fichier existe déjà
      if (existingFiles.includes(file.name)) {
        toast.warning(`"${file.name}" déjà ajouté. Ignoré.`);
        continue;
      }
      
      if (file.size > maxSize) {
        toast.error(`"${file.name}" trop volumineux (max 10MB)`);
        return;
      }
      const ext = file.name.split('.').pop().toLowerCase();
      if (!allowedExtensions.includes(ext)) {
        toast.error(`"${file.name}" format non supporté`);
        return;
      }
    }

    setIsUploading(true);
    const uploadedFiles = [...existingFiles]; // Copier les fichiers existants
    let combinedText = existingText; // Commencer avec le texte existant

    try {
      // Upload des NOUVEAUX fichiers en parallèle pour plus de rapidité
      const uploadPromises = newFiles.map(async (file, index) => {
        const formData = new FormData();
        formData.append('file', file);

        const response = await axios.post(`${API}/upload-file`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          // Timeout augmenté pour PDFs complexes (optimisé backend = extraction plus rapide)
          timeout: 60000,  // 60 secondes (réduit car backend optimisé)
          // Désactiver la mise en cache
          validateStatus: (status) => status < 500
        });

        // Log complet pour debug
        console.log('Upload response COMPLET:', response);
        console.log('Response data:', response.data);
        console.log('Response status:', response.status);

        // Vérifier le status HTTP
        if (response.status !== 200) {
          throw new Error(`Erreur HTTP ${response.status}: ${response.data?.detail || 'Erreur serveur'}`);
        }

        // Vérifier que response.data existe
        if (!response.data) {
          throw new Error('Réponse vide du serveur');
        }

        // Log pour debug
        console.log('Upload response:', {
          filename: response.data.filename,
          has_text: !!response.data.extracted_text,
          text_length: response.data.text_length,
          text_preview: response.data.extracted_text?.substring(0, 50)
        });

        // Vérifier que extracted_text n'est pas undefined ou vide
        if (!response.data.extracted_text) {
          throw new Error(`Texte extrait vide ou undefined. Réponse complète: ${JSON.stringify(response.data)}`);
        }

        return {
          name: response.data.filename,
          extracted_text: response.data.extracted_text,
          text_length: response.data.text_length
        };
      });

      // Attendre tous les uploads
      const results = await Promise.all(uploadPromises);
      
      // Ajouter les nouveaux textes extraits
      results.forEach((result, index) => {
        uploadedFiles.push(result.name);
        const docNumber = existingFiles.length + index + 1;
        combinedText += `\n\n=== DOCUMENT ${docNumber}: ${result.name} ===\n${result.extracted_text}\n`;
      });

      setUploadedFile({
        name: uploadedFiles.length > 1 
          ? `${uploadedFiles.length} fichiers` 
          : uploadedFiles[0],
        names: uploadedFiles,
        extracted_text: combinedText.trim(),
        text_length: combinedText.length,
        count: uploadedFiles.length
      });

      // Message différent si ajout ou premier upload
      if (existingFiles.length > 0) {
        toast.success(`✅ +${newFiles.length} fichier(s) ajouté(s) ! Total: ${uploadedFiles.length} fichiers.`);
      } else {
        toast.success(`📎 ${uploadedFiles.length} fichier(s) analysé(s) ! Posez votre question.`);
      }
      
    } catch (error) {
      console.error('Erreur upload:', error);
      
      // Message d'erreur détaillé selon le type d'erreur
      let errorMessage = 'Erreur lors de l\'analyse du fichier';
      
      if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
        errorMessage = '⏱️ Timeout: Le PDF prend trop de temps à traiter. Seules les 50 premières pages sont extraites. Si votre PDF fait plus de 50 pages, l\'extraction partielle devrait fonctionner.';
      } else if (error.response?.status === 413) {
        errorMessage = 'Fichier trop volumineux. Maximum 10MB par fichier.';
      } else if (error.response?.status === 500 && error.response?.data?.detail?.includes('corrompu')) {
        errorMessage = '❌ PDF corrompu ou protégé. Essayez de l\'ouvrir dans Adobe Reader et de l\'exporter à nouveau.';
      } else if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      }
      
      toast.error(errorMessage, { duration: 5000 });
      
      // IMPORTANT: Réinitialiser l'état pour permettre de nouveaux uploads
      setUploadedFile(null);
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
        // Log pour debug
        console.log('Sending to analyze-file:', {
          question: messageToSend,
          filename: uploadedFile.name,
          has_text: !!uploadedFile.extracted_text,
          text_length: uploadedFile.extracted_text?.length,
          text_preview: uploadedFile.extracted_text?.substring(0, 100)
        });

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

  const downloadDocument = async (content, title, format) => {
    // Utiliser une variable locale au lieu de setIsLoading pour éviter le re-render des messages
    const toastId = toast.loading(`Génération du ${format.toUpperCase()} en cours...`);
    
    try {
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
      
      toast.success(`Document ${format.toUpperCase()} téléchargé avec succès !`, { id: toastId });
      
    } catch (error) {
      console.error('Erreur téléchargement:', error);
      toast.error('Erreur lors du téléchargement du document', { id: toastId });
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

  // Fonctions pour gérer l'historique des conversations (cloud)
  const handleNewConversation = () => {
    setMessages([]);
    setCurrentConversationId(null);
    setConversationHistory([]); // Reset la mémoire
    setSessionId(Date.now().toString());
    setUploadedFile(null);
    toast.success('Nouvelle conversation démarrée');
  };

  const handleSelectConversation = async (convId) => {
    try {
      const conv = await ConversationService.getConversation(convId);
      
      if (!conv) {
        toast.error('Conversation non trouvée');
        return;
      }
      
      // Charger les messages de la conversation
      const loadedMessages = (conv.messages || []).map((msg, index) => ({
        id: Date.now() + index,
        message: msg.content,
        isUser: msg.role === 'user',
        type: activeTab,
        image_base64: msg.image_base64,
        timestamp: new Date(msg.timestamp)
      }));
      
      // Charger l'historique pour la mémoire
      const history = (conv.messages || []).map(msg => ({
        role: msg.role,
        content: msg.content
      }));
      
      setMessages(loadedMessages);
      setConversationHistory(history);
      setCurrentConversationId(convId);
      setIsSidebarOpen(false);
      toast.success(`Conversation "${conv.title}" chargée`);
    } catch (error) {
      console.error('Erreur chargement conversation:', error);
      toast.error('Erreur lors du chargement de la conversation');
    }
  };

  const handleDeleteConversation = async (convId) => {
    await ConversationService.deleteConversation(convId);
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
    setIsLicenseAdmin(false);
    setMessages([]);
    setConversationHistory([]);
    setCurrentConversationId(null);
    setShowAuthModal(true);
    toast.success('Déconnexion réussie');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-white to-blue-50">
      <Toaster position="top-right" />
      
      {/* Boutons flottants en mode plein écran - NIVEAU RACINE */}
      {isChatFullscreen && (
        <div className="fixed top-6 right-6 z-[9999] flex flex-row gap-3">
          <Button
            variant="outline"
            size="default"
            onClick={() => setShowCorrectionModal(true)}
            className="bg-white hover:bg-red-50 border-red-300 shadow-xl px-4 py-2"
            title="Corriger un texte d'élève"
          >
            <span className="flex items-center gap-2 text-red-600">
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 20h9"/>
                <path d="M16.376 3.622a1 1 0 0 1 3.002 3.002L7.368 18.635a2 2 0 0 1-.855.506l-2.872.838a.5.5 0 0 1-.62-.62l.838-2.872a2 2 0 0 1 .506-.854z"/>
              </svg>
              Corriger
            </span>
          </Button>
          <Button
            variant="outline"
            size="default"
            onClick={handlePrintChat}
            className="bg-white hover:bg-blue-50 border-blue-300 shadow-xl px-4 py-2"
            title="Imprimer la conversation"
          >
            <span className="flex items-center gap-2 text-blue-600">
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="6 9 6 2 18 2 18 9"></polyline>
                <path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"></path>
                <rect x="6" y="14" width="12" height="8"></rect>
              </svg>
              Imprimer
            </span>
          </Button>
          <Button
            size="default"
            onClick={() => setIsChatFullscreen(false)}
            className="bg-orange-500 hover:bg-orange-600 text-white shadow-xl px-4 py-2"
          >
            <span className="flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="4 14 10 14 10 20"></polyline>
                <polyline points="20 10 14 10 14 4"></polyline>
                <line x1="14" y1="10" x2="21" y2="3"></line>
                <line x1="3" y1="21" x2="10" y2="14"></line>
              </svg>
              Réduire
            </span>
          </Button>
        </div>
      )}
      
      {/* Modal d'authentification */}
      <AuthModal 
        isOpen={showAuthModal}
        onClose={() => setShowAuthModal(false)}
        onSuccess={(user) => {
          handleAuthSuccess(user);
          // Vérifier le statut d'admin de licence après connexion
          const token = localStorage.getItem('etienne_token');
          if (token) checkLicenseAdmin(token);
        }}
      />
      
      {/* Panneau d'administration (Super Admin) */}
      {showAdminPanel && (
        <AdminPanel 
          onClose={() => setShowAdminPanel(false)}
        />
      )}
      
      {/* Panneau d'administration de licence */}
      {showLicenseAdminPanel && (
        <LicenseAdminPanel 
          onClose={() => setShowLicenseAdminPanel(false)}
        />
      )}
      
      {/* Page de démonstration des symboles mathématiques */}
      {showMathSymbols && (
        <MathSymbolsDemo 
          onClose={() => setShowMathSymbols(false)}
        />
      )}

      {/* Modal de correction de texte MEQ */}
      <TextCorrectionModal
        open={showCorrectionModal}
        onClose={() => setShowCorrectionModal(false)}
        onSubmit={handleCorrectionSubmit}
        apiUrl={API}
      />

      {/* Modal Nouveautés */}
      <Nouveautes
        open={showNouveautes}
        onClose={() => setShowNouveautes(false)}
        isAdmin={isAdmin}
      />
      
      {/* Sidebar d'historique */}
      <ConversationSidebar
        isOpen={isSidebarOpen}
        onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
        currentConversationId={currentConversationId}
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
        onDeleteConversation={handleDeleteConversation}
        currentUser={currentUser}
      />
      
      {/* Header */}
      <HeaderNav
        currentUser={currentUser}
        isAdmin={isAdmin}
        isLicenseAdmin={isLicenseAdmin}
        setShowAuthModal={setShowAuthModal}
        setShowAdminPanel={setShowAdminPanel}
        setShowLicenseAdminPanel={setShowLicenseAdminPanel}
        setShowNouveautes={setShowNouveautes}
        setShowProfileModal={setShowProfileModal}
        handleLogout={handleLogout}
      />

      {/* Affichage du quota en position fixe */}
      <div className="fixed bottom-4 right-4 z-40 w-64">
        <QuotaDisplay />
      </div>

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
            Étienne, votre assistant IA pour <span className="text-transparent bg-clip-text bg-gradient-to-r from-orange-500 to-blue-600">le personnel scolaire</span>
          </h2>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto leading-relaxed">
            Créez des plans de cours, générez des documents professionnels, trouvez des ressources adaptées au programme québécois. Pour enseignants, TES, orthopédagogues, direction, secrétariat et tout le personnel scolaire.
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
          <div className={isChatFullscreen ? "fixed inset-0 z-50 bg-white p-4 overflow-auto" : "lg:col-span-3 relative"}>
            
            {/* Banner de réveil du serveur */}
            {serverWaking && (
              <div className="mb-3 p-3 bg-amber-50 border border-amber-200 rounded-lg flex items-center gap-3" data-testid="server-waking-banner">
                <div className="w-5 h-5 border-2 border-amber-500 border-t-transparent rounded-full animate-spin flex-shrink-0"></div>
                <div>
                  <p className="text-sm font-medium text-amber-800">Le serveur se réveille...</p>
                  <p className="text-xs text-amber-600">Première connexion du jour. Cela peut prendre quelques secondes.</p>
                </div>
              </div>
            )}

            <Card className={`bg-white/90 backdrop-blur-sm border-orange-100 flex flex-col ${isChatFullscreen ? 'h-full' : 'min-h-[600px]'}`}>
              <CardHeader className="pb-2 relative">
                {/* Boutons en haut à droite - mode normal seulement */}
                {!isChatFullscreen && (
                  <div className="absolute top-2 right-2 z-10 flex gap-2">
                    {/* Bouton Corriger un texte */}
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setShowCorrectionModal(true)}
                      className="bg-white hover:bg-red-50 border-red-200"
                      title="Corriger un texte d'élève (protocole MEQ)"
                      data-testid="open-correction-modal-btn"
                    >
                      <span className="flex items-center gap-1 text-red-600">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M12 20h9"/>
                          <path d="M16.376 3.622a1 1 0 0 1 3.002 3.002L7.368 18.635a2 2 0 0 1-.855.506l-2.872.838a.5.5 0 0 1-.62-.62l.838-2.872a2 2 0 0 1 .506-.854z"/>
                        </svg>
                        <span className="hidden sm:inline">Corriger un texte</span>
                      </span>
                    </Button>

                    {/* Bouton Imprimer */}
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handlePrintChat}
                      className="bg-white hover:bg-blue-50 border-blue-200"
                      title="Imprimer la conversation"
                    >
                      <span className="flex items-center gap-1 text-blue-600">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <polyline points="6 9 6 2 18 2 18 9"></polyline>
                          <path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"></path>
                          <rect x="6" y="14" width="12" height="8"></rect>
                        </svg>
                        <span className="hidden sm:inline">Imprimer</span>
                      </span>
                    </Button>
                  
                    {/* Bouton plein écran */}
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setIsChatFullscreen(true)}
                      className="bg-white hover:bg-orange-50 border-orange-200"
                    >
                      <span className="flex items-center gap-1 text-orange-600">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <polyline points="15 3 21 3 21 9"></polyline>
                          <polyline points="9 21 3 21 3 15"></polyline>
                          <line x1="21" y1="3" x2="14" y2="10"></line>
                          <line x1="3" y1="21" x2="10" y2="14"></line>
                        </svg>
                        <span className="hidden sm:inline">Plein écran</span>
                      </span>
                    </Button>
                  </div>
                )}
                
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
                <div 
                  data-testid="chat-scroll-area"
                  className="flex-1 mb-4 border-2 border-orange-200 rounded-lg overflow-y-auto overflow-x-hidden chat-scroll-container" 
                  style={{ 
                    maxHeight: isChatFullscreen ? 'calc(100vh - 350px)' : '550px',
                  }}
                >
                  <div className="space-y-4 p-4">
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
                      messages.map((msg, index) => (
                        <ChatMessage
                          key={msg.id}
                          msg={msg}
                          prevMsg={messages[index - 1]}
                          downloadDocument={downloadDocument}
                          isLoading={isLoading}
                        />
                      ))
                    )}
                    {isLoading && <LoadingIndicator />}
                    <div ref={messagesEndRef} />
                  </div>
                </div>
                
                <Separator className="mb-4" />
                
                {/* Zone d'upload de fichier */}
                {uploadedFile && (
                  <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-blue-600">📎</span>
                        <div>
                          <p className="text-sm font-medium text-blue-800">
                            {uploadedFile.count > 1 
                              ? `${uploadedFile.count} fichiers chargés` 
                              : uploadedFile.name}
                          </p>
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
                    {uploadedFile.names && uploadedFile.names.length > 1 && (
                      <div className="text-xs text-blue-700 mt-2 border-t border-blue-200 pt-2">
                        {uploadedFile.names.map((name, i) => (
                          <div key={i} className="truncate">• {name}</div>
                        ))}
                      </div>
                    )}
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
                      <label className="relative cursor-pointer" title="Joindre un ou plusieurs fichiers (max 5)">
                        <input
                          type="file"
                          onChange={handleFileUpload}
                          accept=".pdf,.docx,.doc,.txt,.xlsx,.xls,.csv,.pptx"
                          className="hidden"
                          disabled={isUploading}
                          multiple
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
                    
                    {/* Fichiers uploadés - Afficher la liste */}
                    {uploadedFile && !isUploading && (
                      <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium text-green-800">
                              📎 {uploadedFile.count || 1} fichier(s) prêt(s)
                            </span>
                          </div>
                          <button
                            onClick={() => {
                              setUploadedFile(null);
                              toast.info('Fichiers effacés');
                            }}
                            className="text-xs text-red-600 hover:text-red-800 font-medium"
                          >
                            ✕ Tout effacer
                          </button>
                        </div>
                        <div className="text-xs text-green-700 mt-1">
                          {uploadedFile.names ? uploadedFile.names.join(', ') : uploadedFile.name}
                        </div>
                        <div className="text-xs text-green-600 mt-1">
                          💡 Cliquez sur 📎 pour ajouter plus de fichiers (max 5)
                        </div>
                      </div>
                    )}
                    
                    {/* Indicateur de progression d'upload amélioré */}
                    {isUploading && (
                      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <div className="animate-spin h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full"></div>
                            <span className="text-sm font-medium text-blue-800">Analyse en cours...</span>
                          </div>
                          <button
                            onClick={() => {
                              setIsUploading(false);
                              toast.info('Upload annulé');
                            }}
                            className="text-xs text-blue-600 hover:text-blue-800"
                          >
                            Annuler
                          </button>
                        </div>
                        <div className="w-full bg-blue-200 rounded-full h-2 overflow-hidden">
                          <div className="bg-blue-600 h-2 rounded-full animate-pulse" style={{width: '100%'}}></div>
                        </div>
                        <div className="text-xs text-blue-600 mt-1">
                          Ajout de nouveaux fichiers aux {uploadedFile?.count || 0} existant(s)...
                        </div>
                      </div>
                    )}
                    
                    {/* Info formats supportés */}
                    <div className="text-xs text-gray-500 flex items-center gap-2">
                      <span>📎 Formats: PDF, Word, Excel, PowerPoint, TXT, CSV</span>
                      <span>•</span>
                      <span>Max: 10MB/fichier</span>
                      <span>•</span>
                      <span>Jusqu'à 5 fichiers</span>
                    </div>
                  </form>
                )}
              </CardContent>
            </Card>
          </div>
          
          {/* Features Section - extracted component */}
          <FeatureSection 
            messageTypes={messageTypes}
            setActiveTab={setActiveTab}
            setShowCorrectionModal={setShowCorrectionModal}
          />
        </div>
        )}
      </div>
      
      {/* Modal Profil Utilisateur */}
      {showProfileModal && (
        <ProfileModal 
          onClose={() => setShowProfileModal(false)} 
          API={API}
          onEmailChanged={(newToken) => {
            localStorage.setItem('token', newToken);
            setShowProfileModal(false);
            const payload = JSON.parse(atob(newToken.split('.')[1]));
            toast.success(`Email changé avec succès. Nouveau: ${payload.email}`);
          }}
        />
      )}
    </div>
  );
}

export default App;
