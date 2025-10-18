import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

function App() {
  const [activeSection, setActiveSection] = useState('chat');
  const [activeTab, setActiveTab] = useState('Je veux');
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  
  // État pour le réveil du backend
  const [backendStatus, setBackendStatus] = useState('checking');
  
  // États pour analyse complète
  const [analyseText, setAnalyseText] = useState('');
  const [analyseResult, setAnalyseResult] = useState(null);
  const [analyseLoading, setAnalyseLoading] = useState(false);
  
  // États pour upload
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [fileQuestion, setFileQuestion] = useState('');
  const [fileQuestionLoading, setFileQuestionLoading] = useState(false);
  const [fileQuestionResult, setFileQuestionResult] = useState(null);

  const tabs = ['Je veux', 'Je recherche', 'Sources fiables', 'Activités éducatives'];

  // Fonction pour réveiller le backend
  const wakeBackend = async () => {
    setBackendStatus('waking');
    let attempts = 0;
    const maxAttempts = 3;

    const pingBackend = async () => {
      try {
        const response = await axios.get(`${BACKEND_URL}/api/`, { timeout: 10000 });
        if (response.status === 200) {
          setBackendStatus('awake');
          return true;
        }
      } catch (error) {
        console.log(`Tentative ${attempts + 1}/${maxAttempts} échouée`);
        return false;
      }
    };

    while (attempts < maxAttempts) {
      const success = await pingBackend();
      if (success) {
        return;
      }
      attempts++;
      if (attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 5000));
      }
    }

    setBackendStatus('awake');
  };

  useEffect(() => {
    wakeBackend();
  }, []);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = {
      role: 'user',
      content: input
    };

    setMessages([...messages, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post(`${BACKEND_URL}/api/chat`, {
        message: input,
        message_type: activeTab
      });

      const assistantMessage = {
        role: 'assistant',
        content: response.data.response || response.data.message,
        sources: response.data.sources,
        trust_score: response.data.trust_score,
        messageId: Date.now()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Erreur:', error);
      const errorMessage = {
        role: 'assistant',
        content: 'Désolé, une erreur est survenue. Veuillez réessayer.',
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleDownload = async (format, messageId) => {
    const message = messages.find(m => m.messageId === messageId);
    if (!message) return;

    try {
      const response = await axios.post(`${BACKEND_URL}/api/generate-document`, {
        content: message.content,
        title: "Document Étienne",
        format: format
      }, {
        responseType: 'blob'
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `etienne-document.${format}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error('Erreur téléchargement:', error);
      alert('Erreur lors du téléchargement');
    }
  };

  const handleAnalyseComplete = async () => {
    if (!analyseText.trim()) return;
    setAnalyseLoading(true);

    try {
      const response = await axios.post(`${BACKEND_URL}/api/analyze-text`, {
        text: analyseText
      });
      setAnalyseResult(response.data);
    } catch (error) {
      console.error('Erreur analyse complète:', error);
      alert('Erreur lors de l\'analyse');
    } finally {
      setAnalyseLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setUploadFile(file);
    setUploadLoading(true);
    setFileQuestionResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${BACKEND_URL}/api/upload-file`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setUploadResult({
        filename: response.data.filename,
        file_type: file.type,
        content: response.data.extracted_text
      });
    } catch (error) {
      console.error('Erreur upload:', error);
      alert('Erreur lors de l\'upload du document');
    } finally {
      setUploadLoading(false);
    }
  };

  const handleFileQuestion = async () => {
    if (!fileQuestion.trim() || !uploadResult) return;
    
    setFileQuestionLoading(true);

    try {
      const response = await axios.post(`${BACKEND_URL}/api/analyze-file`, {
        question: fileQuestion,
        extracted_text: uploadResult.content,
        filename: uploadResult.filename,
        message_type: "je_veux"
      });
      
      setFileQuestionResult({
        question: fileQuestion,
        response: response.data.response
      });
      setFileQuestion('');
    } catch (error) {
      console.error('Erreur analyse fichier:', error);
      alert('Erreur lors de l\'analyse du fichier');
    } finally {
      setFileQuestionLoading(false);
    }
  };

  if (backendStatus === 'checking' || backendStatus === 'waking') {
    return (
      <div className="app-container">
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '100vh',
          textAlign: 'center',
          padding: '20px'
        }}>
          <div style={{
            background: 'white',
            borderRadius: '20px',
            padding: '60px 40px',
            boxShadow: '0 20px 60px rgba(0,0,0,0.2)',
            maxWidth: '500px'
          }}>
            <h1 style={{ color: '#667eea', fontSize: '3rem', marginBottom: '20px' }}>É Étienne</h1>
            <div style={{
              width: '60px',
              height: '60px',
              border: '6px solid #f0f0f0',
              borderTop: '6px solid #667eea',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite',
              margin: '30px auto'
            }}></div>
            <p style={{ color: '#666', fontSize: '1.2rem', marginTop: '20px' }}>
              {backendStatus === 'checking' ? 'Connexion au serveur...' : 'Réveil du serveur en cours...'}
            </p>
            <p style={{ color: '#999', fontSize: '0.9rem', marginTop: '10px' }}>
              Cela peut prendre 30 à 60 secondes
            </p>
          </div>
          <style>{`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
          `}</style>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <header className="header">
        <h1>É Étienne</h1>
        <p>Assistant IA pour les étudiants québécois fourni par le Collège Champagneur</p>
      </header>

      <div className="main-nav">
        <button 
          className={`nav-btn ${activeSection === 'chat' ? 'active' : ''}`}
          onClick={() => setActiveSection('chat')}
        >
          💬 Chat
        </button>
        <button 
          className={`nav-btn ${activeSection === 'analyse' ? 'active' : ''}`}
          onClick={() => setActiveSection('analyse')}
        >
          📊 Analyse
        </button>
        <button 
          className={`nav-btn ${activeSection === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveSection('upload')}
        >
          📄 Upload
        </button>
      </div>

      {activeSection === 'chat' && (
        <div className="chat-container">
          <div className="tabs">
            {tabs.map(tab => (
              <button
                key={tab}
                className={`tab ${activeTab === tab ? 'active' : ''}`}
                onClick={() => setActiveTab(tab)}
              >
                {tab}
              </button>
            ))}
          </div>

          <div className="messages-area">
            {messages.length === 0 && (
              <div style={{ textAlign: 'center', color: '#999', padding: '40px' }}>
                <h3>Bonjour! 👋</h3>
                <p>Je suis Étienne, votre assistant IA. Comment puis-je vous aider aujourd'hui?</p>
              </div>
            )}

            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.role}`}>
                <div className="message-label">
                  {msg.role === 'user' ? 'Vous' : 'Étienne'}
                </div>
                <div className="message-content">
                  {msg.content}
                </div>
                {msg.trust_score && msg.trust_score >= 0.9 && (
                  <div className="trust-badge">
                    ✓ Sources fiables ({Math.round(msg.trust_score * 100)}%)
                  </div>
                )}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="sources">
                    <strong>Sources:</strong> {msg.sources.slice(0, 3).join(', ')}
                  </div>
                )}
                {msg.role === 'assistant' && msg.content.length > 200 && (
                  <div className="download-buttons">
                    <button onClick={() => handleDownload('pdf', msg.messageId)} className="download-btn">📄 PDF</button>
                    <button onClick={() => handleDownload('docx', msg.messageId)} className="download-btn">📝 Word</button>
                    <button onClick={() => handleDownload('pptx', msg.messageId)} className="download-btn">📊 PowerPoint</button>
                  </div>
                )}
              </div>
            ))}

            {loading && (
              <div className="loading">
                Étienne réfléchit...
              </div>
            )}
          </div>

          <div className="input-area">
            <input
              type="text"
              className="chat-input"
              placeholder="Posez votre question à Étienne..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={loading}
            />
            <button
              className="send-button"
              onClick={handleSend}
              disabled={loading || !input.trim()}
            >
              {loading ? '⏳' : 'Envoyer'}
            </button>
          </div>
        </div>
      )}

      {activeSection === 'analyse' && (
        <div className="tool-container">
          <h2>📊 Analyse Complète</h2>
          <p>Analyse complète incluant détection IA, plagiat et statistiques du texte.</p>
          <textarea
            className="tool-textarea"
            placeholder="Collez le texte à analyser ici..."
            value={analyseText}
            onChange={(e) => setAnalyseText(e.target.value)}
            rows={10}
          />
          <button 
            className="tool-button"
            onClick={handleAnalyseComplete}
            disabled={analyseLoading || !analyseText.trim()}
          >
            {analyseLoading ? '📊 Analyse...' : '📊 Analyser'}
          </button>

          {analyseResult && (
            <div className="result-box">
              <h3>Résultat de l'analyse complète</h3>
              
              <div className="result-item">
                <strong>📝 Statistiques:</strong>
              </div>
              <div className="result-item">
                <strong>Langue:</strong> {analyseResult.language === 'fr' ? 'Français' : 'Anglais'}
              </div>
              <div className="result-item">
                <strong>Nombre de mots:</strong> {analyseResult.overall_assessment?.word_count || 0}
              </div>

              {analyseResult.ai_detection && (
                <>
                  <div className="result-item" style={{ marginTop: '20px' }}>
                    <strong>🤖 Détection IA:</strong>
                  </div>
                  <div className={`result-item ${analyseResult.ai_detection.is_likely_ai ? 'danger' : 'success'}`}>
                    <strong>Verdict:</strong> {analyseResult.ai_detection.is_likely_ai ? '⚠️ Probablement généré par IA' : '✅ Probablement écrit par un humain'}
                  </div>
                  <div className="result-item">
                    <strong>Probabilité IA:</strong> {Math.round(analyseResult.ai_detection.ai_probability * 100)}%
                  </div>
                  <div className="result-item">
                    <strong>Confiance:</strong> {analyseResult.ai_detection.confidence}
                  </div>
                </>
              )}

              {analyseResult.plagiarism_check && (
                <>
                  <div className="result-item" style={{ marginTop: '20px' }}>
                    <strong>📋 Détection Plagiat:</strong>
                  </div>
                  <div className={`result-item ${analyseResult.plagiarism_check.is_suspicious ? 'danger' : 'success'}`}>
                    <strong>Verdict:</strong> {analyseResult.plagiarism_check.is_suspicious ? '⚠️ Risque de plagiat' : '✅ Contenu original'}
                  </div>
                  <div className="result-item">
                    <strong>Score de risque:</strong> {Math.round(analyseResult.plagiarism_check.plagiarism_risk * 100)}%
                  </div>
                  <div className="result-item">
                    <strong>Niveau:</strong> {analyseResult.plagiarism_check.risk_level}
                  </div>
                </>
              )}

              {analyseResult.overall_assessment?.recommendations && (
                <div className="result-item" style={{ marginTop: '20px' }}>
                  <strong>💡 Recommandations:</strong>
                  <ul>
                    {analyseResult.overall_assessment.recommendations.map((rec, i) => (
                      <li key={i}>{rec}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {activeSection === 'upload' && (
        <div className="tool-container">
          <h2>📄 Upload de Document</h2>
          <p>Uploadez un document (PDF, Word, Excel, PowerPoint) pour l'analyser et poser des questions.</p>
          
          <div className="upload-area">
            <input
              type="file"
              id="file-upload"
              className="file-input"
              onChange={handleFileUpload}
              accept=".pdf,.doc,.docx,.xls,.xlsx,.ppt,.pptx,.txt,.csv"
            />
            <label htmlFor="file-upload" className="file-label">
              📁 Choisir un fichier
            </label>
            {uploadFile && <p className="file-name">Fichier: {uploadFile.name}</p>}
          </div>

          {uploadLoading && <div className="loading">Upload en cours...</div>}

          {uploadResult && (
            <div className="result-box">
              <h3>Document analysé</h3>
              <div className="result-item">
                <strong>Nom:</strong> {uploadResult.filename}
              </div>
              <div className="result-item">
                <strong>Type:</strong> {uploadResult.file_type}
              </div>
              <div className="result-item">
                <strong>Contenu extrait:</strong>
                <pre className="extracted-content">{uploadResult.content.substring(0, 500)}...</pre>
              </div>

              <div style={{ marginTop: '30px', padding: '20px', background: '#f0f9ff', borderRadius: '10px' }}>
                <h4 style={{ color: '#667eea', marginBottom: '15px' }}>💬 Poser une question sur ce document</h4>
                <div className="input-area">
                  <input
                    type="text"
                    className="chat-input"
                    placeholder="Ex: Résume ce document, Quels sont les points clés?, etc."
                    value={fileQuestion}
                    onChange={(e) => setFileQuestion(e.target.value)}
                    onKeyPress={(e) => {
                      if (e.key === 'Enter' && !fileQuestionLoading && fileQuestion.trim()) {
                        handleFileQuestion();
                      }
                    }}
                    disabled={fileQuestionLoading}
                  />
                  <button
                    className="send-button"
                    onClick={handleFileQuestion}
                    disabled={fileQuestionLoading || !fileQuestion.trim()}
                  >
                    {fileQuestionLoading ? '⏳' : 'Analyser'}
                  </button>
                </div>
              </div>

              {fileQuestionResult && (
                <div style={{ marginTop: '20px' }}>
                  <div className="message user" style={{ marginLeft: '0', marginBottom: '15px' }}>
                    <div className="message-label">Votre question</div>
                    <div className="message-content">{fileQuestionResult.question}</div>
                  </div>
                  <div className="message assistant" style={{ marginRight: '0' }}>
                    <div className="message-label">Réponse d'Étienne</div>
                    <div className="message-content">{fileQuestionResult.response}</div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
