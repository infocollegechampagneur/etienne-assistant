import React, { useState } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

function App() {
  const [activeTab, setActiveTab] = useState('Je veux');
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const tabs = ['Je veux', 'Je recherche', 'Sources fiables', 'Activités éducatives'];

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
        tab: activeTab
      });

      const assistantMessage = {
        role: 'assistant',
        content: response.data.response || response.data.message,
        sources: response.data.sources,
        trust_score: response.data.trust_score
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

  return (
    <div className=\"app-container\">
      <header className=\"header\">
        <h1>É Étienne</h1>
        <p>Assistant IA pour les étudiants québécois fourni par le Collège Champagneur</p>
      </header>

      <div className=\"chat-container\">
        <div className=\"tabs\">
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

        <div className=\"messages-area\">
          {messages.length === 0 && (
            <div style={{ textAlign: 'center', color: '#999', padding: '40px' }}>
              <h3>Bonjour! 👋</h3>
              <p>Je suis Étienne, votre assistant IA. Comment puis-je vous aider aujourd'hui?</p>
            </div>
          )}

          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.role}`}>
              <div className=\"message-label\">
                {msg.role === 'user' ? 'Vous' : 'Étienne'}
              </div>
              <div className=\"message-content\">
                {msg.content}
              </div>
              {msg.trust_score && msg.trust_score >= 0.9 && (
                <div className=\"trust-badge\">
                  ✓ Sources fiables ({Math.round(msg.trust_score * 100)}%)
                </div>
              )}
              {msg.sources && msg.sources.length > 0 && (
                <div className=\"sources\">
                  <strong>Sources:</strong> {msg.sources.slice(0, 3).join(', ')}
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className=\"loading\">
              Étienne réfléchit...
            </div>
          )}
        </div>

        <div className=\"input-area\">
          <input
            type=\"text\"
            className=\"chat-input\"
            placeholder=\"Posez votre question à Étienne...\"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={loading}
          />
          <button
            className=\"send-button\"
            onClick={handleSend}
            disabled={loading || !input.trim()}
          >
            {loading ? '⏳' : 'Envoyer'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
