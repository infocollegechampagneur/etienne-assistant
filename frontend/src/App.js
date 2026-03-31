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
import ConversationService from './ConversationService';
import AuthModal from './components/AuthModal';
import AdminPanel from './AdminPanel';
import LicenseAdminPanel from './LicenseAdminPanel';
import QuotaDisplay from './QuotaDisplay';
import MathSymbolsDemo from './MathSymbolsDemo';
import TextCorrectionModal from './TextCorrectionModal';
import { formatMessage, cleanMessageForExport } from './utils/formatMessage';
import 'katex/dist/katex.min.css';
import katex from 'katex';
import { InlineMath, BlockMath } from 'react-katex';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Composant pour rendre le contenu avec support LaTeX
const MessageContent = ({ html }) => {
  const containerRef = useRef(null);
  const [renderedHtml, setRenderedHtml] = useState(html);
  
  useEffect(() => {
    // Trouver et remplacer les spans LaTeX par le rendu KaTeX
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = html;
    
    // Traiter les éléments latex-inline et latex-block
    const latexElements = tempDiv.querySelectorAll('.latex-inline, .latex-block');
    
    if (latexElements.length > 0) {
      latexElements.forEach((el) => {
        const latexContent = decodeURIComponent(el.getAttribute('data-latex') || '');
        if (!latexContent) return;
        
        const isBlock = el.classList.contains('latex-block');
        
        try {
          const rendered = katex.renderToString(latexContent, {
            throwOnError: false,
            displayMode: isBlock,
            trust: true
          });
          el.innerHTML = rendered;
          el.className = isBlock ? 'katex-block-container' : 'katex-inline-container';
        } catch (error) {
          console.warn('Erreur rendu LaTeX:', error, latexContent);
        }
      });
      
      setRenderedHtml(tempDiv.innerHTML);
    }
  }, [html]);
  
  return (
    <div 
      ref={containerRef}
      className="text-sm leading-relaxed formatted-message" 
      dangerouslySetInnerHTML={{__html: renderedHtml}}
    />
  );
};

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
      const scrollArea = messagesEndRef.current.closest('[data-radix-scroll-area-viewport]');
      if (scrollArea) {
        scrollArea.scrollTop = scrollArea.scrollHeight;
      } else {
        messagesEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }
  };

  // Fonction pour imprimer le contenu du chat
  // Fonction pour convertir le LaTeX en texte lisible pour l'impression
  const cleanLatexForPrint = (text) => {
    if (!text) return '';
    
    let cleaned = text;
    
    // Traiter les fractions imbriquées d'abord (plusieurs passes)
    for (let i = 0; i < 5; i++) {
      cleaned = cleaned.replace(/\\frac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}/g, '($1/$2)');
    }
    
    // Convertir les racines carrées avec imbrications possibles
    cleaned = cleaned.replace(/\\sqrt\[(\d+)\]\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}/g, '∜($2)'); // racine n-ième
    cleaned = cleaned.replace(/\\sqrt\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}/g, '√($1)');
    
    // Supprimer les délimiteurs LaTeX APRÈS les conversions
    cleaned = cleaned.replace(/\$\$/g, '');
    cleaned = cleaned.replace(/\$/g, '');
    cleaned = cleaned.replace(/\\\[/g, '');
    cleaned = cleaned.replace(/\\\]/g, '');
    cleaned = cleaned.replace(/\\\(/g, '');
    cleaned = cleaned.replace(/\\\)/g, '');
    
    // Convertir les puissances: x^{2} → x² ou x^2 → x²
    cleaned = cleaned.replace(/\^{0}/g, '⁰');
    cleaned = cleaned.replace(/\^0(?![0-9])/g, '⁰');
    cleaned = cleaned.replace(/\^{1}/g, '¹');
    cleaned = cleaned.replace(/\^1(?![0-9])/g, '¹');
    cleaned = cleaned.replace(/\^{2}/g, '²');
    cleaned = cleaned.replace(/\^2(?![0-9])/g, '²');
    cleaned = cleaned.replace(/\^{3}/g, '³');
    cleaned = cleaned.replace(/\^3(?![0-9])/g, '³');
    cleaned = cleaned.replace(/\^{4}/g, '⁴');
    cleaned = cleaned.replace(/\^4(?![0-9])/g, '⁴');
    cleaned = cleaned.replace(/\^{5}/g, '⁵');
    cleaned = cleaned.replace(/\^{6}/g, '⁶');
    cleaned = cleaned.replace(/\^{7}/g, '⁷');
    cleaned = cleaned.replace(/\^{8}/g, '⁸');
    cleaned = cleaned.replace(/\^{9}/g, '⁹');
    cleaned = cleaned.replace(/\^{n}/g, 'ⁿ');
    cleaned = cleaned.replace(/\^n(?![a-z])/g, 'ⁿ');
    cleaned = cleaned.replace(/\^{([^}]*)}/g, '^($1)');
    
    // Convertir les indices: x_{1} → x₁
    cleaned = cleaned.replace(/_{0}/g, '₀');
    cleaned = cleaned.replace(/_0(?![0-9])/g, '₀');
    cleaned = cleaned.replace(/_{1}/g, '₁');
    cleaned = cleaned.replace(/_1(?![0-9])/g, '₁');
    cleaned = cleaned.replace(/_{2}/g, '₂');
    cleaned = cleaned.replace(/_2(?![0-9])/g, '₂');
    cleaned = cleaned.replace(/_{3}/g, '₃');
    cleaned = cleaned.replace(/_3(?![0-9])/g, '₃');
    cleaned = cleaned.replace(/_{4}/g, '₄');
    cleaned = cleaned.replace(/_{5}/g, '₅');
    cleaned = cleaned.replace(/_{6}/g, '₆');
    cleaned = cleaned.replace(/_{7}/g, '₇');
    cleaned = cleaned.replace(/_{8}/g, '₈');
    cleaned = cleaned.replace(/_{9}/g, '₉');
    cleaned = cleaned.replace(/_{([^}]*)}/g, '_($1)');
    
    // Symboles mathématiques courants
    cleaned = cleaned.replace(/\\times/g, '×');
    cleaned = cleaned.replace(/\\div/g, '÷');
    cleaned = cleaned.replace(/\\pm/g, '±');
    cleaned = cleaned.replace(/\\mp/g, '∓');
    cleaned = cleaned.replace(/\\le(?:q)?/g, '≤');
    cleaned = cleaned.replace(/\\ge(?:q)?/g, '≥');
    cleaned = cleaned.replace(/\\neq/g, '≠');
    cleaned = cleaned.replace(/\\ne(?![a-z])/g, '≠');
    cleaned = cleaned.replace(/\\approx/g, '≈');
    cleaned = cleaned.replace(/\\sim/g, '~');
    cleaned = cleaned.replace(/\\infty/g, '∞');
    cleaned = cleaned.replace(/\\pi/g, 'π');
    cleaned = cleaned.replace(/\\alpha/g, 'α');
    cleaned = cleaned.replace(/\\beta/g, 'β');
    cleaned = cleaned.replace(/\\gamma/g, 'γ');
    cleaned = cleaned.replace(/\\delta/g, 'δ');
    cleaned = cleaned.replace(/\\epsilon/g, 'ε');
    cleaned = cleaned.replace(/\\theta/g, 'θ');
    cleaned = cleaned.replace(/\\lambda/g, 'λ');
    cleaned = cleaned.replace(/\\mu/g, 'μ');
    cleaned = cleaned.replace(/\\sigma/g, 'σ');
    cleaned = cleaned.replace(/\\omega/g, 'ω');
    cleaned = cleaned.replace(/\\Delta/g, 'Δ');
    cleaned = cleaned.replace(/\\Sigma/g, 'Σ');
    cleaned = cleaned.replace(/\\Omega/g, 'Ω');
    cleaned = cleaned.replace(/\\sum/g, 'Σ');
    cleaned = cleaned.replace(/\\prod/g, 'Π');
    cleaned = cleaned.replace(/\\int/g, '∫');
    cleaned = cleaned.replace(/\\rightarrow/g, '→');
    cleaned = cleaned.replace(/\\leftarrow/g, '←');
    cleaned = cleaned.replace(/\\Rightarrow/g, '⇒');
    cleaned = cleaned.replace(/\\Leftarrow/g, '⇐');
    cleaned = cleaned.replace(/\\leftrightarrow/g, '↔');
    cleaned = cleaned.replace(/\\therefore/g, '∴');
    cleaned = cleaned.replace(/\\because/g, '∵');
    cleaned = cleaned.replace(/\\forall/g, '∀');
    cleaned = cleaned.replace(/\\exists/g, '∃');
    cleaned = cleaned.replace(/\\in/g, '∈');
    cleaned = cleaned.replace(/\\notin/g, '∉');
    cleaned = cleaned.replace(/\\subset/g, '⊂');
    cleaned = cleaned.replace(/\\supset/g, '⊃');
    cleaned = cleaned.replace(/\\cup/g, '∪');
    cleaned = cleaned.replace(/\\cap/g, '∩');
    cleaned = cleaned.replace(/\\emptyset/g, '∅');
    cleaned = cleaned.replace(/\\angle/g, '∠');
    cleaned = cleaned.replace(/\\perp/g, '⊥');
    cleaned = cleaned.replace(/\\parallel/g, '∥');
    cleaned = cleaned.replace(/\\degree/g, '°');
    cleaned = cleaned.replace(/\\circ/g, '°');
    
    // Nettoyer les espaces LaTeX
    cleaned = cleaned.replace(/\\,/g, ' ');
    cleaned = cleaned.replace(/\\;/g, ' ');
    cleaned = cleaned.replace(/\\:/g, ' ');
    cleaned = cleaned.replace(/\\!/g, '');
    cleaned = cleaned.replace(/\\ /g, ' ');
    cleaned = cleaned.replace(/\\quad/g, '  ');
    cleaned = cleaned.replace(/\\qquad/g, '    ');
    cleaned = cleaned.replace(/~/g, ' ');
    
    // Nettoyer les commandes LaTeX restantes
    cleaned = cleaned.replace(/\\text\{([^}]*)\}/g, '$1');
    cleaned = cleaned.replace(/\\textbf\{([^}]*)\}/g, '$1');
    cleaned = cleaned.replace(/\\textit\{([^}]*)\}/g, '$1');
    cleaned = cleaned.replace(/\\mathrm\{([^}]*)\}/g, '$1');
    cleaned = cleaned.replace(/\\mathbf\{([^}]*)\}/g, '$1');
    cleaned = cleaned.replace(/\\mathit\{([^}]*)\}/g, '$1');
    cleaned = cleaned.replace(/\\mathsf\{([^}]*)\}/g, '$1');
    cleaned = cleaned.replace(/\\overline\{([^}]*)\}/g, '$1');
    cleaned = cleaned.replace(/\\underline\{([^}]*)\}/g, '$1');
    cleaned = cleaned.replace(/\\left/g, '');
    cleaned = cleaned.replace(/\\right/g, '');
    cleaned = cleaned.replace(/\\big/g, '');
    cleaned = cleaned.replace(/\\Big/g, '');
    cleaned = cleaned.replace(/\\cdot/g, '·');
    cleaned = cleaned.replace(/\\ldots/g, '...');
    cleaned = cleaned.replace(/\\dots/g, '...');
    cleaned = cleaned.replace(/\\vdots/g, '⋮');
    cleaned = cleaned.replace(/\\ddots/g, '⋱');
    cleaned = cleaned.replace(/\\begin\{[^}]*\}/g, '');
    cleaned = cleaned.replace(/\\end\{[^}]*\}/g, '');
    cleaned = cleaned.replace(/\\hline/g, '');
    cleaned = cleaned.replace(/&/g, ' | ');
    cleaned = cleaned.replace(/\\\\/g, ' ');
    
    // Supprimer les backslashs restants devant les commandes inconnues
    cleaned = cleaned.replace(/\\([a-zA-Z]+)/g, '$1');
    
    // Nettoyer les accolades vides et simples
    cleaned = cleaned.replace(/\{\}/g, '');
    // Plusieurs passes pour les accolades imbriquées
    for (let i = 0; i < 3; i++) {
      cleaned = cleaned.replace(/\{([^{}]*)\}/g, '$1');
    }
    
    // Nettoyer les parenthèses vides ou redondantes
    cleaned = cleaned.replace(/\(\)/g, '');
    cleaned = cleaned.replace(/\(\s*\)/g, '');
    
    // Nettoyer les espaces multiples
    cleaned = cleaned.replace(/\s+/g, ' ');
    
    return cleaned.trim();
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
      // Envoyer le message au backend avec l'historique pour la mémoire
      const response = await axios.post(`${API}/chat`, {
        message: messageToSend,
        message_type: activeTab,
        session_id: sessionId,
        conversation_id: convId,
        conversation_history: updatedHistory.slice(-10)
      });

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
      <header className="bg-white/90 backdrop-blur-md border-b border-orange-100 sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-orange-500 to-blue-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-lg">É</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Étienne</h1>
                <p className="text-sm text-gray-600">Assistant IA pour les membres du personnel scolaire québécois</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                ✅ Programme québécois
              </Badge>
              
              {/* Boutons Auth/Admin */}
              {currentUser ? (
                <>
                  <Badge variant="outline" className="bg-purple-50 text-purple-700 border-purple-200">
                    👤 {currentUser.full_name}
                  </Badge>
                  {isLicenseAdmin && !isAdmin && (
                    <Button 
                      size="sm" 
                      onClick={() => setShowLicenseAdminPanel(true)}
                      className="bg-blue-600 hover:bg-blue-700"
                    >
                      🏢 Ma licence
                    </Button>
                  )}
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
                    onClick={() => setShowProfileModal(true)}
                    className="border-blue-200 hover:bg-blue-50"
                  >
                    👤 Profil
                  </Button>
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
                <ScrollArea 
                  className="flex-1 mb-4 border-2 border-orange-200 rounded-lg" 
                  style={{ 
                    maxHeight: isChatFullscreen ? 'calc(100vh - 350px)' : '550px',
                    overflow: 'auto',
                    scrollbarWidth: 'auto',
                    scrollbarColor: '#f97316 #fef3e2'
                  }}
                >
                  <div 
                    className="space-y-4 pr-2" 
                    style={{
                      height: isChatFullscreen ? 'calc(100vh - 350px)' : '550px',
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
                                  <MessageContent html={formatMessage(msg.message)} />
                                )}
                                
                                {/* Affichage des images générées (multiples ou unique) */}
                                {(msg.images && msg.images.length > 0) ? (
                                  <div className="mt-3 grid grid-cols-2 md:grid-cols-3 gap-3">
                                    {msg.images.map((imgBase64, idx) => (
                                      <div key={idx} className="border rounded-lg p-2 bg-white">
                                        <img 
                                          src={`data:image/png;base64,${imgBase64}`} 
                                          alt={`Diagramme ${idx + 1}`} 
                                          className="rounded max-w-full h-auto"
                                        />
                                        <a 
                                          href={`data:image/png;base64,${imgBase64}`}
                                          download={`angle_${idx + 1}_${Date.now()}.png`}
                                          className="inline-block mt-1 text-xs bg-green-100 hover:bg-green-200 text-green-700 px-2 py-1 rounded transition-colors w-full text-center"
                                        >
                                          💾 Télécharger
                                        </a>
                                      </div>
                                    ))}
                                  </div>
                                ) : msg.image_base64 && (
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
            {/* Carte Corriger un texte */}
            <Card 
              className="bg-gradient-to-br from-red-50 to-orange-50 border-red-200 hover:shadow-lg transition-shadow cursor-pointer ring-2 ring-red-100" 
              onClick={() => setShowCorrectionModal(true)}
              data-testid="correction-feature-card"
            >
              <CardContent className="p-6 text-center">
                <div className="text-3xl mb-3">
                  <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mx-auto text-red-600"><path d="M12 20h9"/><path d="M16.376 3.622a1 1 0 0 1 3.002 3.002L7.368 18.635a2 2 0 0 1-.855.506l-2.872.838a.5.5 0 0 1-.62-.62l.838-2.872a2 2 0 0 1 .506-.854z"/></svg>
                </div>
                <h4 className="font-semibold text-gray-900 mb-2">Corriger un texte</h4>
                <p className="text-sm text-gray-600">Correction conforme aux grilles MEQ avec pondérations personnalisables</p>
              </CardContent>
            </Card>
          </div>
        </div>
        
        {/* Educational Image */}
        {/* Section Fonctionnalités */}
        <div className="lg:col-span-4 mt-12 space-y-8">
          
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
                <h4 className="font-semibold text-gray-800 mb-2">Exemples de demandes :</h4>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• "Résume ce manuel scolaire pour créer un plan de cours"</li>
                  <li>• "Extrais les concepts clés de ce chapitre"</li>
                  <li>• "Crée des questions d'examen basées sur ce document"</li>
                  <li>• "Analyse ce plan d'intervention pour un élève HDAA"</li>
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
                <h4 className="font-semibold text-gray-800 mb-2">Exemples par rôle :</h4>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• <strong>Enseignant:</strong> "Plan de cours sur la Révolution tranquille"</li>
                  <li>• <strong>TES:</strong> "Crée un plan d'intervention pour élève TDAH"</li>
                  <li>• <strong>Orthopédagogue:</strong> "Évaluation diagnostique en lecture Sec 1"</li>
                  <li>• <strong>Direction:</strong> "Projet éducatif pour conseil d'établissement"</li>
                  <li>• <strong>Secrétariat:</strong> "Lettre aux parents - activité parascolaire"</li>
                  <li>• <strong>Travailleur social:</strong> "Modèle rapport d'évaluation psychosociale"</li>
                </ul>
              </div>
            </div>
          </div>
          
          {/* Section Ce qu'Étienne peut faire pour vous */}
          <div className="bg-gradient-to-r from-orange-50 to-yellow-50 rounded-xl p-6 border border-orange-200">
            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              🎓 Ce qu'Étienne peut faire pour vous
            </h3>
            
            {/* Enseignants */}
            <div className="mb-6">
              <h4 className="font-semibold text-orange-700 mb-2 flex items-center gap-2">👨‍🏫 Enseignants</h4>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-2 text-sm text-gray-600">
                <span>• Plans de cours (PFEQ)</span>
                <span>• Examens avec corrigés</span>
                <span>• SAÉ et activités</span>
                <span>• Grilles d'évaluation critériées</span>
                <span>• Exercices différenciés</span>
                <span>• Présentations PowerPoint</span>
              </div>
            </div>
            
            {/* Intervenants */}
            <div className="mb-6">
              <h4 className="font-semibold text-blue-700 mb-2 flex items-center gap-2">👩‍⚕️ TES, Orthopédagogues, Intervenants psychosociaux</h4>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-2 text-sm text-gray-600">
                <span>• Plans d'intervention (PI)</span>
                <span>• Fiches d'observation</span>
                <span>• Stratégies TDAH, TSA, dyslexie</span>
                <span>• Rapports de suivi</span>
                <span>• Évaluations diagnostiques</span>
                <span>• Outils de gestion de crise</span>
              </div>
            </div>
            
            {/* Travail social */}
            <div className="mb-6">
              <h4 className="font-semibold text-purple-700 mb-2 flex items-center gap-2">🤝 Travailleurs sociaux</h4>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-2 text-sm text-gray-600">
                <span>• Évaluations psychosociales</span>
                <span>• Rapports DPJ (LPJ)</span>
                <span>• Références ressources</span>
                <span>• Documentation signalements</span>
                <span>• Suivis confidentiels</span>
                <span>• Interventions de groupe</span>
              </div>
            </div>
            
            {/* Administration */}
            <div className="mb-6">
              <h4 className="font-semibold text-green-700 mb-2 flex items-center gap-2">🏢 Direction et Administration</h4>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-2 text-sm text-gray-600">
                <span>• Projets éducatifs</span>
                <span>• Plans stratégiques</span>
                <span>• Politiques institutionnelles</span>
                <span>• Communications officielles</span>
                <span>• Présentations CA</span>
                <span>• Rapports annuels</span>
              </div>
            </div>
            
            {/* Secrétariat */}
            <div className="mb-6">
              <h4 className="font-semibold text-pink-700 mb-2 flex items-center gap-2">📋 Secrétariat et TOS</h4>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-2 text-sm text-gray-600">
                <span>• Lettres officielles</span>
                <span>• Communications parents</span>
                <span>• Formulaires (absences, autorisations)</span>
                <span>• Procès-verbaux</span>
                <span>• Grilles-horaires</span>
                <span>• Procédures administratives</span>
              </div>
            </div>
            
            {/* Autres postes */}
            <div>
              <h4 className="font-semibold text-teal-700 mb-2 flex items-center gap-2">🔧 Autres membres du personnel</h4>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-2 text-sm text-gray-600">
                <span>• <strong>Tech. loisir:</strong> Planification événements</span>
                <span>• <strong>Surveillants:</strong> Protocoles, rapports incidents</span>
                <span>• <strong>Tech. labo:</strong> Procédures sécurité (SIMDUT)</span>
                <span>• <strong>Coord. comm.:</strong> Infolettres, médias sociaux</span>
                <span>• <strong>Finance:</strong> Budgets, rapports financiers</span>
                <span>• <strong>Admissions:</strong> Formulaires, statistiques</span>
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
            <p className="text-gray-600 mt-4 italic">Étienne - Assistant IA pour tout le personnel scolaire québécois</p>
          </div>
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
            // Recharger les infos utilisateur
            const payload = JSON.parse(atob(newToken.split('.')[1]));
            toast.success(`Email changé avec succès. Nouveau: ${payload.email}`);
          }}
        />
      )}
    </div>
  );
}

// Composant Modal de Profil pour changer l'email
const ProfileModal = ({ onClose, API, onEmailChanged }) => {
  const [activeTab, setActiveTab] = useState('email'); // 'email' ou 'password'
  const [newEmail, setNewEmail] = useState('');
  const [password, setPassword] = useState('');
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentEmail, setCurrentEmail] = useState('');
  
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      try {
        const payload = JSON.parse(atob(token.split('.')[1]));
        setCurrentEmail(payload.email || '');
      } catch (e) {
        console.error('Erreur décodage token:', e);
      }
    }
  }, []);
  
  const handleChangeEmail = async (e) => {
    e.preventDefault();
    if (!newEmail || !password) {
      toast.error('Veuillez remplir tous les champs');
      return;
    }
    if (newEmail === currentEmail) {
      toast.error('Le nouvel email doit être différent');
      return;
    }
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await axios.post(
        `${API}/api/auth/change-email`,
        { new_email: newEmail, password: password },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      toast.success(response.data.message);
      onEmailChanged(response.data.new_token);
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Erreur lors du changement');
    } finally {
      setLoading(false);
    }
  };
  
  const handleChangePassword = async (e) => {
    e.preventDefault();
    if (!currentPassword || !newPassword || !confirmPassword) {
      toast.error('Veuillez remplir tous les champs');
      return;
    }
    if (newPassword !== confirmPassword) {
      toast.error('Les mots de passe ne correspondent pas');
      return;
    }
    if (newPassword.length < 6) {
      toast.error('Le mot de passe doit contenir au moins 6 caractères');
      return;
    }
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await axios.post(
        `${API}/api/auth/change-password`,
        { current_password: currentPassword, new_password: newPassword },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      toast.success(response.data.message);
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Erreur lors du changement');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <Card className="w-full max-w-md">
        <CardHeader>
          <div className="flex justify-between items-center">
            <CardTitle className="flex items-center gap-2">👤 Mon Profil</CardTitle>
            <button onClick={onClose} className="text-2xl hover:text-red-600">✕</button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="mb-4 p-3 bg-gray-50 rounded-lg">
            <p className="text-sm text-gray-600">Email actuel:</p>
            <p className="font-semibold">{currentEmail}</p>
          </div>
          
          {/* Tabs */}
          <div className="flex border-b mb-4">
            <button
              onClick={() => setActiveTab('email')}
              className={`flex-1 py-2 text-sm font-medium ${activeTab === 'email' ? 'border-b-2 border-blue-600 text-blue-600' : 'text-gray-500'}`}
            >
              ✉️ Changer email
            </button>
            <button
              onClick={() => setActiveTab('password')}
              className={`flex-1 py-2 text-sm font-medium ${activeTab === 'password' ? 'border-b-2 border-blue-600 text-blue-600' : 'text-gray-500'}`}
            >
              🔐 Changer mot de passe
            </button>
          </div>
          
          {activeTab === 'email' ? (
            <form onSubmit={handleChangeEmail} className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Nouvel email</label>
                <Input type="email" value={newEmail} onChange={(e) => setNewEmail(e.target.value)} placeholder="nouveau@email.com" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Mot de passe actuel</label>
                <Input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Confirmez votre identité" />
              </div>
              <Button type="submit" className="w-full" disabled={loading}>
                {loading ? '⏳ Modification...' : '✏️ Changer mon email'}
              </Button>
            </form>
          ) : (
            <form onSubmit={handleChangePassword} className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Mot de passe actuel</label>
                <Input type="password" value={currentPassword} onChange={(e) => setCurrentPassword(e.target.value)} placeholder="Votre mot de passe actuel" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Nouveau mot de passe</label>
                <Input type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} placeholder="Min. 6 caractères" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Confirmer le nouveau mot de passe</label>
                <Input type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} placeholder="Répétez le nouveau mot de passe" />
              </div>
              <Button type="submit" className="w-full" disabled={loading}>
                {loading ? '⏳ Modification...' : '🔐 Changer mon mot de passe'}
              </Button>
            </form>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default App;
