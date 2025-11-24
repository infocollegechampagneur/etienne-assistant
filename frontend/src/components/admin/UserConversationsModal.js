/**
 * UserConversationsModal.js
 * Modal pour afficher l'historique des conversations d'un utilisateur
 */

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { ScrollArea } from '../ui/scroll-area';
import { toast } from 'sonner';

const UserConversationsModal = ({ user, onClose }) => {
  const [conversations, setConversations] = useState([]);
  const [loading, setLoading] = useState(false);

  const API = process.env.REACT_APP_BACKEND_URL;
  const token = localStorage.getItem('etienne_token');

  const axiosConfig = {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  };

  useEffect(() => {
    loadConversations();
  }, []);

  const loadConversations = async () => {
    setLoading(true);
    try {
      const response = await axios.get(
        `${API}/api/admin/users/${user.id}/conversations`,
        axiosConfig
      );
      setConversations(response.data.conversations);
    } catch (error) {
      toast.error('Erreur chargement conversations');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <Card className="w-full max-w-4xl max-h-[90vh] flex flex-col">
        <CardHeader>
          <div className="flex justify-between items-center">
            <CardTitle>
              💬 Historique de {user.full_name}
            </CardTitle>
            <button
              onClick={onClose}
              className="text-2xl hover:text-red-600 transition"
            >
              ✕
            </button>
          </div>
          <div className="text-sm text-gray-600">
            📧 {user.email} • 🏢 {user.organization}
          </div>
        </CardHeader>
        <CardContent className="flex-1 overflow-hidden">
          {loading ? (
            <div className="text-center p-8">Chargement...</div>
          ) : conversations.length === 0 ? (
            <div className="text-center p-8 text-gray-500">
              Aucune conversation enregistrée pour cet utilisateur.
            </div>
          ) : (
            <ScrollArea className="h-[60vh] pr-4">
              <div className="space-y-4">
                {conversations.map((conv, index) => (
                  <div
                    key={index}
                    className={`p-4 rounded-lg ${
                      conv.role === 'user'
                        ? 'bg-blue-50 border-l-4 border-blue-500'
                        : 'bg-green-50 border-l-4 border-green-500'
                    }`}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <span className="font-semibold">
                        {conv.role === 'user' ? '👤 Utilisateur' : '🤖 Étienne'}
                      </span>
                      <span className="text-xs text-gray-500">
                        {new Date(conv.timestamp).toLocaleString('fr-CA', {
                          dateStyle: 'short',
                          timeStyle: 'short'
                        })}
                      </span>
                    </div>
                    <div className="text-sm whitespace-pre-wrap">
                      {conv.content}
                    </div>
                    {conv.session_id && (
                      <div className="text-xs text-gray-400 mt-2">
                        Session: {conv.session_id}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </ScrollArea>
          )}
          <div className="mt-4 flex justify-end">
            <Button onClick={onClose}>Fermer</Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default UserConversationsModal;
