/**
 * ConversationHistory.js
 * Système complet d'historique de conversations pour Étienne
 * Stockage 100% local (localStorage) pour la confidentialité
 */

import React, { useState, useEffect } from 'react';

// Utilitaires pour localStorage
const STORAGE_KEY = 'etienne_conversations';
const SETTINGS_KEY = 'etienne_history_settings';

export const ConversationHistory = {
  
  // ==================== GESTION DU STOCKAGE ====================
  
  getAllConversations: () => {
    try {
      const data = localStorage.getItem(STORAGE_KEY);
      return data ? JSON.parse(data) : {};
    } catch (error) {
      console.error('Erreur lecture conversations:', error);
      return {};
    }
  },

  saveConversations: (conversations) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
    } catch (error) {
      console.error('Erreur sauvegarde conversations:', error);
    }
  },

  getSettings: () => {
    try {
      const data = localStorage.getItem(SETTINGS_KEY);
      return data ? JSON.parse(data) : {
        maxConversations: 100,
        retentionDays: 30,
        autoCleanup: true
      };
    } catch (error) {
      return { maxConversations: 100, retentionDays: 30, autoCleanup: true };
    }
  },

  saveSettings: (settings) => {
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    } catch (error) {
      console.error('Erreur sauvegarde settings:', error);
    }
  },

  // ==================== GESTION DES CONVERSATIONS ====================

  createConversation: (firstMessage) => {
    const conversations = ConversationHistory.getAllConversations();
    const id = `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Générer un titre depuis le premier message
    const title = firstMessage.substring(0, 50) + (firstMessage.length > 50 ? '...' : '');
    
    const newConversation = {
      id,
      title,
      messages: [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      isPinned: false,
      isFavorite: false,
      tags: [],
      notes: '',
      summary: ''
    };
    
    conversations[id] = newConversation;
    ConversationHistory.saveConversations(conversations);
    
    return id;
  },

  addMessage: (conversationId, message) => {
    const conversations = ConversationHistory.getAllConversations();
    
    if (!conversations[conversationId]) {
      console.error('Conversation introuvable:', conversationId);
      return;
    }
    
    conversations[conversationId].messages.push({
      ...message,
      timestamp: new Date().toISOString()
    });
    
    conversations[conversationId].updatedAt = new Date().toISOString();
    
    // Générer un résumé après 5 messages
    if (conversations[conversationId].messages.length >= 5 && 
        conversations[conversationId].messages.length % 5 === 0) {
      conversations[conversationId].summary = ConversationHistory.generateSummary(
        conversations[conversationId].messages
      );
    }
    
    ConversationHistory.saveConversations(conversations);
  },

  updateConversation: (conversationId, updates) => {
    const conversations = ConversationHistory.getAllConversations();
    
    if (!conversations[conversationId]) return;
    
    conversations[conversationId] = {
      ...conversations[conversationId],
      ...updates,
      updatedAt: new Date().toISOString()
    };
    
    ConversationHistory.saveConversations(conversations);
  },

  deleteConversation: (conversationId) => {
    const conversations = ConversationHistory.getAllConversations();
    delete conversations[conversationId];
    ConversationHistory.saveConversations(conversations);
  },

  deleteAll: () => {
    localStorage.removeItem(STORAGE_KEY);
  },

  // ==================== GÉNÉRATION DE RÉSUMÉ ====================

  generateSummary: (messages) => {
    // Extraire les sujets principaux des messages
    const topics = [];
    const keywords = ['comment', 'pourquoi', 'qu\'est-ce', 'expliquer', 'résoudre', 'calculer'];
    
    messages.forEach(msg => {
      if (msg.role === 'user') {
        const lowerMsg = msg.content.toLowerCase();
        keywords.forEach(keyword => {
          if (lowerMsg.includes(keyword)) {
            topics.push(msg.content.substring(0, 100));
          }
        });
      }
    });
    
    return topics.slice(0, 3).join('; ') || 'Discussion générale';
  },

  // ==================== RECHERCHE ET FILTRAGE ====================

  searchConversations: (query) => {
    const conversations = ConversationHistory.getAllConversations();
    const lowerQuery = query.toLowerCase();
    
    return Object.values(conversations).filter(conv => {
      return conv.title.toLowerCase().includes(lowerQuery) ||
             conv.notes.toLowerCase().includes(lowerQuery) ||
             conv.tags.some(tag => tag.toLowerCase().includes(lowerQuery)) ||
             conv.messages.some(msg => msg.content.toLowerCase().includes(lowerQuery));
    });
  },

  filterByTag: (tag) => {
    const conversations = ConversationHistory.getAllConversations();
    return Object.values(conversations).filter(conv => conv.tags.includes(tag));
  },

  getFavorites: () => {
    const conversations = ConversationHistory.getAllConversations();
    return Object.values(conversations).filter(conv => conv.isFavorite);
  },

  getPinned: () => {
    const conversations = ConversationHistory.getAllConversations();
    return Object.values(conversations).filter(conv => conv.isPinned);
  },

  // ==================== GROUPEMENT PAR DATE ====================

  groupByDate: (conversations) => {
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    const weekAgo = new Date(today);
    weekAgo.setDate(weekAgo.getDate() - 7);
    const monthAgo = new Date(today);
    monthAgo.setMonth(monthAgo.getMonth() - 1);

    const groups = {
      "Aujourd'hui": [],
      "Hier": [],
      "Cette semaine": [],
      "Ce mois": [],
      "Plus ancien": []
    };

    conversations.forEach(conv => {
      const convDate = new Date(conv.updatedAt);
      
      if (convDate >= today) {
        groups["Aujourd'hui"].push(conv);
      } else if (convDate >= yesterday) {
        groups["Hier"].push(conv);
      } else if (convDate >= weekAgo) {
        groups["Cette semaine"].push(conv);
      } else if (convDate >= monthAgo) {
        groups["Ce mois"].push(conv);
      } else {
        groups["Plus ancien"].push(conv);
      }
    });

    return groups;
  },

  // ==================== EXPORT ====================

  exportToPDF: async (conversationId) => {
    // Cette fonction sera appelée depuis App.js qui a accès au backend
    return conversationId;
  },

  exportToText: (conversationId) => {
    const conversations = ConversationHistory.getAllConversations();
    const conv = conversations[conversationId];
    
    if (!conv) return null;
    
    let text = `Conversation: ${conv.title}\n`;
    text += `Date: ${new Date(conv.createdAt).toLocaleString('fr-FR')}\n`;
    text += `Tags: ${conv.tags.join(', ')}\n\n`;
    text += '='.repeat(50) + '\n\n';
    
    conv.messages.forEach(msg => {
      text += `${msg.role === 'user' ? 'Utilisateur' : 'Étienne'}: ${msg.content}\n\n`;
    });
    
    return text;
  },

  downloadTextFile: (conversationId) => {
    const text = ConversationHistory.exportToText(conversationId);
    if (!text) return;
    
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `conversation_${conversationId}_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  },

  // ==================== NETTOYAGE AUTOMATIQUE ====================

  cleanup: () => {
    const settings = ConversationHistory.getSettings();
    if (!settings.autoCleanup) return;
    
    const conversations = ConversationHistory.getAllConversations();
    const now = new Date();
    const cutoffDate = new Date(now);
    cutoffDate.setDate(cutoffDate.getDate() - settings.retentionDays);
    
    // Supprimer les conversations trop anciennes (sauf épinglées/favorites)
    Object.entries(conversations).forEach(([id, conv]) => {
      const convDate = new Date(conv.updatedAt);
      if (convDate < cutoffDate && !conv.isPinned && !conv.isFavorite) {
        delete conversations[id];
      }
    });
    
    // Limiter le nombre total de conversations
    const sortedConvs = Object.values(conversations)
      .sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
    
    if (sortedConvs.length > settings.maxConversations) {
      const toDelete = sortedConvs
        .slice(settings.maxConversations)
        .filter(conv => !conv.isPinned && !conv.isFavorite);
      
      toDelete.forEach(conv => {
        delete conversations[conv.id];
      });
    }
    
    ConversationHistory.saveConversations(conversations);
  },

  // ==================== PARTAGE ====================

  generateShareLink: (conversationId) => {
    // Créer un lien encodé en base64 (conversation peut être grosse)
    const conversations = ConversationHistory.getAllConversations();
    const conv = conversations[conversationId];
    
    if (!conv) return null;
    
    // Simplifier la conversation pour le partage
    const shareData = {
      title: conv.title,
      messages: conv.messages.slice(0, 20), // Limiter à 20 messages
      tags: conv.tags
    };
    
    const encoded = btoa(encodeURIComponent(JSON.stringify(shareData)));
    return `${window.location.origin}?shared=${encoded}`;
  }
};

export default ConversationHistory;
