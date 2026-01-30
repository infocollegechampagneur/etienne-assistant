/**
 * ConversationSidebar.js
 * Sidebar de l'historique des conversations - Style ChatGPT
 * Utilise ConversationService pour la synchronisation cloud
 */

import React, { useState, useEffect } from 'react';
import ConversationService from './ConversationService';

const ConversationSidebar = ({ 
  isOpen, 
  onToggle, 
  currentConversationId, 
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
  currentUser  // Nouveau prop pour savoir si l'utilisateur est connecté
}) => {
  const [conversations, setConversations] = useState({});
  const [searchQuery, setSearchQuery] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [selectedConv, setSelectedConv] = useState(null);
  const [showActions, setShowActions] = useState(null);
  const [filterMode, setFilterMode] = useState('all'); // all, favorites, pinned
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (currentUser) {
      loadConversations();
    } else {
      setConversations({});
    }
  }, [currentUser]);

  const loadConversations = async () => {
    setIsLoading(true);
    try {
      const convs = await ConversationService.getAllConversations();
      setConversations(convs);
    } catch (error) {
      console.error('Erreur chargement conversations:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearch = (query) => {
    setSearchQuery(query);
  };

  const getFilteredConversations = () => {
    let convList = Object.values(conversations);
    
    // Filtrer par mode
    if (filterMode === 'favorites') {
      convList = convList.filter(c => c.is_favorite);
    } else if (filterMode === 'pinned') {
      convList = convList.filter(c => c.is_pinned);
    }
    
    // Recherche
    if (searchQuery) {
      const lowerQuery = searchQuery.toLowerCase();
      convList = convList.filter(conv =>
        conv.title.toLowerCase().includes(lowerQuery) ||
        (conv.tags && conv.tags.some(tag => tag.toLowerCase().includes(lowerQuery)))
      );
    }
    
    // Trier: épinglées en haut, puis par date
    return convList.sort((a, b) => {
      if (a.is_pinned && !b.is_pinned) return -1;
      if (!a.is_pinned && b.is_pinned) return 1;
      return new Date(b.updated_at) - new Date(a.updated_at);
    });
  };

  const handleToggleFavorite = async (convId, e) => {
    e.stopPropagation();
    const conv = conversations[convId];
    await ConversationService.toggleFavorite(convId, conv.is_favorite);
    loadConversations();
  };

  const handleTogglePin = async (convId, e) => {
    e.stopPropagation();
    const conv = conversations[convId];
    await ConversationService.togglePin(convId, conv.is_pinned);
    loadConversations();
  };

  const handleDelete = async (convId, e) => {
    e.stopPropagation();
    if (window.confirm('Supprimer cette conversation?')) {
      await ConversationService.deleteConversation(convId);
      if (currentConversationId === convId) {
        onNewConversation();
      }
      loadConversations();
    }
  };

  const handleRename = async (convId) => {
    const conv = conversations[convId];
    const newTitle = prompt('Nouveau titre:', conv.title);
    if (newTitle && newTitle.trim()) {
      await ConversationService.renameConversation(convId, newTitle.trim());
      loadConversations();
    }
  };

  const handleExportTxt = (convId, e) => {
    e.stopPropagation();
    const conv = conversations[convId];
    if (conv && conv.messages) {
      let content = `Conversation: ${conv.title}\n`;
      content += `Date: ${new Date(conv.created_at).toLocaleString('fr-CA')}\n`;
      content += `${'='.repeat(50)}\n\n`;
      
      conv.messages.forEach(msg => {
        const role = msg.role === 'user' ? '👤 Vous' : '🤖 Étienne';
        content += `${role}:\n${msg.content}\n\n`;
      });
      
      const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `conversation_${convId}.txt`;
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  const handleShare = (convId, e) => {
    e.stopPropagation();
    // Pour l'instant, copier le titre
    const conv = conversations[convId];
    if (conv) {
      navigator.clipboard.writeText(`Conversation Étienne: ${conv.title}`);
      alert('Titre copié dans le presse-papier!');
    }
  };

  // Grouper les conversations par date
  const groupByDate = (convList) => {
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const yesterday = new Date(today.getTime() - 86400000);
    const lastWeek = new Date(today.getTime() - 7 * 86400000);
    const lastMonth = new Date(today.getTime() - 30 * 86400000);
    
    const groups = {
      "Aujourd'hui": [],
      "Hier": [],
      "Cette semaine": [],
      "Ce mois": [],
      "Plus ancien": []
    };
    
    convList.forEach(conv => {
      const convDate = new Date(conv.updated_at || conv.created_at);
      
      if (convDate >= today) {
        groups["Aujourd'hui"].push(conv);
      } else if (convDate >= yesterday) {
        groups["Hier"].push(conv);
      } else if (convDate >= lastWeek) {
        groups["Cette semaine"].push(conv);
      } else if (convDate >= lastMonth) {
        groups["Ce mois"].push(conv);
      } else {
        groups["Plus ancien"].push(conv);
      }
    });
    
    return groups;
  };

  const groupedConversations = groupByDate(getFilteredConversations());

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

  // Si l'utilisateur n'est pas connecté
  if (!currentUser) {
    return (
      <div className="conversation-sidebar">
        <div className="sidebar-header">
          <button onClick={onToggle} className="sidebar-close-btn">✕</button>
          <h2>💬 Historique</h2>
        </div>
        <div className="sidebar-empty">
          <p>🔐 Connectez-vous pour accéder à votre historique de conversations synchronisé.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="conversation-sidebar">
      {/* Header */}
      <div className="sidebar-header">
        <button onClick={onToggle} className="sidebar-close-btn">✕</button>
        <h2>💬 Historique</h2>
        {isLoading && <span className="loading-indicator">⏳</span>}
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
                    {conv.is_pinned && <span className="pin-icon">📌</span>}
                    <div className="conv-title">{conv.title}</div>
                    {conv.is_favorite && <span className="fav-icon">⭐</span>}
                  </div>
                  
                  {conv.tags && conv.tags.length > 0 && (
                    <div className="conv-tags">
                      {conv.tags.slice(0, 2).map(tag => (
                        <span key={tag} className="tag">{tag}</span>
                      ))}
                    </div>
                  )}
                  
                  <div className="conv-actions">
                    <button
                      onClick={(e) => handleToggleFavorite(conv.id, e)}
                      title={conv.is_favorite ? 'Retirer des favoris' : 'Ajouter aux favoris'}
                    >
                      {conv.is_favorite ? '⭐' : '☆'}
                    </button>
                    <button
                      onClick={(e) => handleTogglePin(conv.id, e)}
                      title={conv.is_pinned ? 'Désépingler' : 'Épingler'}
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

      {/* Footer */}
      <div className="sidebar-footer">
        <button onClick={loadConversations} className="settings-btn">
          🔄 Actualiser
        </button>
      </div>
    </div>
  );
};

export default ConversationSidebar;
