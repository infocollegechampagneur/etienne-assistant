/**
 * ConversationService.js
 * Service pour gérer les conversations synchronisées avec le cloud (MongoDB)
 * Remplace le stockage localStorage par des appels API
 */

import axios from 'axios';

const API = process.env.REACT_APP_BACKEND_URL;

export const ConversationService = {
  
  // ==================== GESTION DES CONVERSATIONS ====================
  
  /**
   * Récupérer toutes les conversations de l'utilisateur
   */
  getAllConversations: async () => {
    try {
      const token = localStorage.getItem('etienne_token');
      if (!token) return {};
      
      const response = await axios.get(`${API}/api/conversations`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      
      if (response.data.success) {
        // Convertir en objet indexé par ID pour compatibilité
        const convObj = {};
        response.data.conversations.forEach(conv => {
          convObj[conv.id] = conv;
        });
        return convObj;
      }
      return {};
    } catch (error) {
      console.error('Erreur récupération conversations:', error);
      return {};
    }
  },

  /**
   * Créer une nouvelle conversation
   */
  createConversation: async (firstMessage) => {
    try {
      const token = localStorage.getItem('etienne_token');
      if (!token) {
        console.warn('Utilisateur non connecté - conversation non sauvegardée');
        return `local_${Date.now()}`;
      }
      
      const title = firstMessage.substring(0, 50) + (firstMessage.length > 50 ? '...' : '');
      
      const response = await axios.post(`${API}/api/conversations`, 
        { title, first_message: firstMessage },
        { headers: { 'Authorization': `Bearer ${token}` } }
      );
      
      if (response.data.success) {
        return response.data.conversation_id;
      }
      return null;
    } catch (error) {
      console.error('Erreur création conversation:', error);
      return null;
    }
  },

  /**
   * Récupérer une conversation avec ses messages
   */
  getConversation: async (conversationId) => {
    try {
      const token = localStorage.getItem('etienne_token');
      if (!token) return null;
      
      const response = await axios.get(`${API}/api/conversations/${conversationId}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      
      if (response.data.success) {
        return response.data.conversation;
      }
      return null;
    } catch (error) {
      console.error('Erreur récupération conversation:', error);
      return null;
    }
  },

  /**
   * Ajouter un message à une conversation
   */
  addMessage: async (conversationId, message) => {
    try {
      const token = localStorage.getItem('etienne_token');
      if (!token || !conversationId || conversationId.startsWith('local_')) {
        return false;
      }
      
      await axios.post(
        `${API}/api/conversations/${conversationId}/messages`,
        message,
        { headers: { 'Authorization': `Bearer ${token}` } }
      );
      
      return true;
    } catch (error) {
      console.error('Erreur ajout message:', error);
      return false;
    }
  },

  /**
   * Mettre à jour une conversation
   */
  updateConversation: async (conversationId, updates) => {
    try {
      const token = localStorage.getItem('etienne_token');
      if (!token) return false;
      
      await axios.put(
        `${API}/api/conversations/${conversationId}`,
        updates,
        { headers: { 'Authorization': `Bearer ${token}` } }
      );
      
      return true;
    } catch (error) {
      console.error('Erreur mise à jour conversation:', error);
      return false;
    }
  },

  /**
   * Supprimer une conversation
   */
  deleteConversation: async (conversationId) => {
    try {
      const token = localStorage.getItem('etienne_token');
      if (!token) return false;
      
      await axios.delete(`${API}/api/conversations/${conversationId}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      
      return true;
    } catch (error) {
      console.error('Erreur suppression conversation:', error);
      return false;
    }
  },

  /**
   * Toggle favoris
   */
  toggleFavorite: async (conversationId, currentState) => {
    return await ConversationService.updateConversation(conversationId, {
      is_favorite: !currentState
    });
  },

  /**
   * Toggle épinglé
   */
  togglePin: async (conversationId, currentState) => {
    return await ConversationService.updateConversation(conversationId, {
      is_pinned: !currentState
    });
  },

  /**
   * Renommer une conversation
   */
  renameConversation: async (conversationId, newTitle) => {
    return await ConversationService.updateConversation(conversationId, {
      title: newTitle
    });
  }
};

export default ConversationService;
