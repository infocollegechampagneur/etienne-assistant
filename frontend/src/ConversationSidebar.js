/**
 * ConversationSidebar.js
 * Sidebar de l'historique des conversations - Style ChatGPT
 */

import React, { useState, useEffect } from 'react';
import ConversationHistory from './ConversationHistory';

const ConversationSidebar = ({ 
  isOpen, 
  onToggle, 
  currentConversationId, 
  onSelectConversation,
  onNewConversation,
  onDeleteConversation
}) => {
  const [conversations, setConversations] = useState({});
  const [searchQuery, setSearchQuery] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [selectedConv, setSelectedConv] = useState(null);
  const [showActions, setShowActions] = useState(null);
  const [settings, setSettings] = useState(ConversationHistory.getSettings());
  const [filterMode, setFilterMode] = useState('all'); // all, favorites, pinned

  useEffect(() => {
    loadConversations();
    // Nettoyage automatique au chargement
    ConversationHistory.cleanup();
  }, []);

  const loadConversations = () => {
    const convs = ConversationHistory.getAllConversations();
    setConversations(convs);
  };

  const handleSearch = (query) => {
    setSearchQuery(query);
  };

  const getFilteredConversations = () => {
    let convList = Object.values(conversations);
    
    // Filtrer par mode
    if (filterMode === 'favorites') {
      convList = convList.filter(c => c.isFavorite);
    } else if (filterMode === 'pinned') {
      convList = convList.filter(c => c.isPinned);
    }
    
    // Recherche
    if (searchQuery) {
      const lowerQuery = searchQuery.toLowerCase();
      convList = convList.filter(conv =>
        conv.title.toLowerCase().includes(lowerQuery) ||
        conv.tags.some(tag => tag.toLowerCase().includes(lowerQuery))
      );
    }
    
    // Trier: épinglées en haut, puis par date
    return convList.sort((a, b) => {
      if (a.isPinned && !b.isPinned) return -1;
      if (!a.isPinned && b.isPinned) return 1;
      return new Date(b.updatedAt) - new Date(a.updatedAt);
    });
  };

  const handleToggleFavorite = (convId, e) => {
    e.stopPropagation();
    const conv = conversations[convId];
    ConversationHistory.updateConversation(convId, {
      isFavorite: !conv.isFavorite
    });
    loadConversations();
  };

  const handleTogglePin = (convId, e) => {
    e.stopPropagation();
    const conv = conversations[convId];
    ConversationHistory.updateConversation(convId, {
      isPinned: !conv.isPinned
    });
    loadConversations();
  };

  const handleDelete = (convId, e) => {
    e.stopPropagation();
    if (window.confirm('Supprimer cette conversation?')) {
      ConversationHistory.deleteConversation(convId);
      if (currentConversationId === convId) {
        onNewConversation();
      }
      loadConversations();
    }
  };

  const handleRename = (convId) => {
    const conv = conversations[convId];
    const newTitle = prompt('Nouveau titre:', conv.title);
    if (newTitle && newTitle.trim()) {
      ConversationHistory.updateConversation(convId, {
        title: newTitle.trim()
      });
      loadConversations();
    }
  };

  const handleExportTxt = (convId, e) => {
    e.stopPropagation();
    ConversationHistory.downloadTextFile(convId);
  };

  const handleShare = (convId, e) => {
    e.stopPropagation();
    const link = ConversationHistory.generateShareLink(convId);
    if (link) {
      navigator.clipboard.writeText(link);
      alert('Lien copié dans le presse-papier!');
    }
  };

  const handleDeleteAll = () => {
    if (window.confirm('Supprimer TOUT l\'historique? Cette action est irréversible!')) {
      ConversationHistory.deleteAll();
      setConversations({});
      onNewConversation();
    }
  };

  const handleSaveSettings = () => {
    ConversationHistory.saveSettings(settings);
    setShowSettings(false);
    ConversationHistory.cleanup();
    loadConversations();
  };

  const groupedConversations = ConversationHistory.groupByDate(getFilteredConversations());

  if (!isOpen) {
    return (
      <button 
        onClick={onToggle}
        className="sidebar-toggle-btn"
        title="Ouvrir l'historique"
      >
        📜
      </button>
    );
  }

  return (
    <div className="conversation-sidebar">
      {/* Header */}
      <div className="sidebar-header">
        <button onClick={onToggle} className="sidebar-close-btn">✕</button>
        <h2>💬 Historique</h2>
      </div>

      {/* Nouveau bouton */}
      <button onClick={onNewConversation} className="new-conversation-btn">
        ➕ Nouvelle conversation
      </button>

      {/* Filtres rapides */}
      <div className="sidebar-filters">
        <button 
          className={`filter-btn ${filterMode === 'all' ? 'active' : ''}`}
          onClick={() => setFilterMode('all')}
        >
          Toutes
        </button>
        <button 
          className={`filter-btn ${filterMode === 'pinned' ? 'active' : ''}`}
          onClick={() => setFilterMode('pinned')}
        >
          📌 Épinglées
        </button>
        <button 
          className={`filter-btn ${filterMode === 'favorites' ? 'active' : ''}`}
          onClick={() => setFilterMode('favorites')}
        >
          ⭐ Favoris
        </button>
      </div>

      {/* Recherche */}
      <div className="sidebar-search">
        <input
          type="text"
          placeholder="🔍 Rechercher..."
          value={searchQuery}
          onChange={(e) => handleSearch(e.target.value)}
        />
      </div>

      {/* Liste des conversations */}
      <div className="conversations-list">
        {Object.entries(groupedConversations).map(([group, convs]) => {
          if (convs.length === 0) return null;
          
          return (
            <div key={group} className="conversation-group">
              <div className="group-title">{group}</div>
              {convs.map(conv => (
                <div
                  key={conv.id}
                  className={`conversation-item ${currentConversationId === conv.id ? 'active' : ''}`}
                  onClick={() => onSelectConversation(conv.id)}
                >
                  <div className="conv-title-row">
                    {conv.isPinned && <span className="pin-icon">📌</span>}
                    <div className="conv-title">{conv.title}</div>
                    {conv.isFavorite && <span className="fav-icon">⭐</span>}
                  </div>
                  
                  {conv.tags.length > 0 && (
                    <div className="conv-tags">
                      {conv.tags.slice(0, 2).map(tag => (
                        <span key={tag} className="tag">{tag}</span>
                      ))}
                    </div>
                  )}
                  
                  <div className="conv-actions">
                    <button
                      onClick={(e) => handleToggleFavorite(conv.id, e)}
                      title={conv.isFavorite ? 'Retirer des favoris' : 'Ajouter aux favoris'}
                    >
                      {conv.isFavorite ? '⭐' : '☆'}
                    </button>
                    <button
                      onClick={(e) => handleTogglePin(conv.id, e)}
                      title={conv.isPinned ? 'Désépingler' : 'Épingler'}
                    >
                      📌
                    </button>
                    <button
                      onClick={() => handleRename(conv.id)}
                      title="Renommer"
                    >
                      ✏️
                    </button>
                    <button
                      onClick={(e) => handleExportTxt(conv.id, e)}
                      title="Exporter en TXT"
                    >
                      📄
                    </button>
                    <button
                      onClick={(e) => handleShare(conv.id, e)}
                      title="Copier lien de partage"
                    >
                      🔗
                    </button>
                    <button
                      onClick={(e) => handleDelete(conv.id, e)}
                      title="Supprimer"
                      className="delete-btn"
                    >
                      🗑️
                    </button>
                  </div>
                </div>
              ))}
            </div>
          );
        })}
        
        {getFilteredConversations().length === 0 && (
          <div className="empty-state">
            <p>Aucune conversation trouvée</p>
            <p className="empty-subtitle">
              {searchQuery ? 'Essayez une autre recherche' : 'Commencez une nouvelle conversation!'}
            </p>
          </div>
        )}
      </div>

      {/* Footer avec paramètres */}
      <div className="sidebar-footer">
        <button onClick={() => setShowSettings(!showSettings)} className="settings-btn">
          ⚙️ Paramètres
        </button>
        <button onClick={handleDeleteAll} className="delete-all-btn">
          🗑️ Tout supprimer
        </button>
      </div>

      {/* Modal Paramètres */}
      {showSettings && (
        <div className="settings-modal">
          <div className="settings-content">
            <h3>⚙️ Paramètres de l'historique</h3>
            
            <div className="setting-item">
              <label>Nombre maximum de conversations:</label>
              <input
                type="number"
                value={settings.maxConversations}
                onChange={(e) => setSettings({...settings, maxConversations: parseInt(e.target.value)})}
                min="10"
                max="1000"
              />
            </div>
            
            <div className="setting-item">
              <label>Durée de conservation (jours):</label>
              <input
                type="number"
                value={settings.retentionDays}
                onChange={(e) => setSettings({...settings, retentionDays: parseInt(e.target.value)})}
                min="1"
                max="365"
              />
            </div>
            
            <div className="setting-item">
              <label>
                <input
                  type="checkbox"
                  checked={settings.autoCleanup}
                  onChange={(e) => setSettings({...settings, autoCleanup: e.target.checked})}
                />
                Nettoyage automatique
              </label>
            </div>
            
            <div className="settings-actions">
              <button onClick={handleSaveSettings} className="btn-primary">Enregistrer</button>
              <button onClick={() => setShowSettings(false)} className="btn-secondary">Annuler</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ConversationSidebar;
