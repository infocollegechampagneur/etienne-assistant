/**
 * BlockedWordsTab.js
 * Onglet Mots Bloqués du panneau admin - NOUVELLE FONCTIONNALITÉ
 */

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Badge } from '../ui/badge';
import { toast } from 'sonner';

const BlockedWordsTab = () => {
  const [words, setWords] = useState([]);
  const [loading, setLoading] = useState(false);
  const [newWord, setNewWord] = useState({
    word: '',
    category: 'custom',
    severity: 'medium',
    is_exception: false
  });

  const API = process.env.REACT_APP_BACKEND_URL;
  const token = localStorage.getItem('etienne_token');

  const axiosConfig = {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  };

  useEffect(() => {
    loadWords();
  }, []);

  const loadWords = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/api/admin/blocked-words`, axiosConfig);
      setWords(response.data.words);
    } catch (error) {
      toast.error('Érreur chargement mots bloqués');
    } finally {
      setLoading(false);
    }
  };

  const addWord = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      await axios.post(`${API}/api/admin/blocked-words`, newWord, axiosConfig);
      toast.success('Mot ajouté avec succès');
      setNewWord({
        word: '',
        category: 'custom',
        severity: 'medium',
        is_exception: false
      });
      loadWords();
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Érreur ajout mot');
    } finally {
      setLoading(false);
    }
  };

  const deleteWord = async (wordId, word) => {
    if (!window.confirm(`Supprimer le mot "${word}" ?`)) {
      return;
    }

    try {
      await axios.delete(`${API}/api/admin/blocked-words/${wordId}`, axiosConfig);
      toast.success('Mot supprimé');
      loadWords();
    } catch (error) {
      toast.error('Érreur suppression');
    }
  };

  const toggleException = async (wordId, currentStatus) => {
    try {
      await axios.put(
        `${API}/api/admin/blocked-words/${wordId}`,
        { is_exception: !currentStatus },
        axiosConfig
      );
      toast.success('Statut modifié');
      loadWords();
    } catch (error) {
      toast.error('Érreur modification');
    }
  };

  const getSeverityColor = (severity) => {
    switch(severity) {
      case 'critical': return 'bg-red-600';
      case 'high': return 'bg-orange-500';
      case 'medium': return 'bg-yellow-500';
      case 'low': return 'bg-green-500';
      default: return 'bg-gray-500';
    }
  };

  const getCategoryLabel = (category) => {
    const labels = {
      'violence': 'Violence',
      'drugs': 'Drogues',
      'illegal': 'Illégal',
      'custom': 'Personnalisé'
    };
    return labels[category] || category;
  };

  return (
    <div className="space-y-6 mt-4">
      {/* Formulaire ajout mot */}
      <Card className="bg-orange-50">
        <CardHeader>
          <CardTitle className="text-lg">⛔ Ajouter un Mot Bloqué</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={addWord} className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <Input
                placeholder="Mot à bloquer"
                value={newWord.word}
                onChange={(e) => setNewWord({...newWord, word: e.target.value})}
                required
              />
              <select
                className="border rounded px-3 py-2"
                value={newWord.category}
                onChange={(e) => setNewWord({...newWord, category: e.target.value})}
              >
                <option value="custom">Personnalisé</option>
                <option value="violence">Violence</option>
                <option value="drugs">Drogues</option>
                <option value="illegal">Illégal</option>
              </select>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <select
                className="border rounded px-3 py-2"
                value={newWord.severity}
                onChange={(e) => setNewWord({...newWord, severity: e.target.value})}
              >
                <option value="low">Faible</option>
                <option value="medium">Moyen</option>
                <option value="high">Élevé</option>
                <option value="critical">Critique</option>
              </select>
              <label className="flex items-center gap-2 px-3 py-2 border rounded bg-white">
                <input
                  type="checkbox"
                  checked={newWord.is_exception}
                  onChange={(e) => setNewWord({...newWord, is_exception: e.target.checked})}
                />
                <span className="text-sm">Exception (autoriser)</span>
              </label>
            </div>
            <Button type="submit" disabled={loading} className="w-full">
              {loading ? 'Ajout...' : 'Ajouter le Mot'}
            </Button>
          </form>
        </CardContent>
      </Card>

      {/* Liste des mots */}
      <div className="space-y-3">
        <h3 className="text-lg font-semibold">Mots Bloqués ({words.length})</h3>
        {loading ? (
          <div className="text-center p-4">Chargement...</div>
        ) : words.length === 0 ? (
          <div className="text-center p-4 text-gray-500">Aucun mot bloqué</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {words.map(word => (
              <Card key={word.id}>
                <CardContent className="p-4">
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="font-bold text-lg">{word.word}</span>
                        <Badge className={getSeverityColor(word.severity)}>
                          {word.severity}
                        </Badge>
                        {word.is_exception && (
                          <Badge className="bg-blue-500">Exception</Badge>
                        )}
                      </div>
                      <div className="text-sm text-gray-600">
                        Catégorie: {getCategoryLabel(word.category)}
                      </div>
                    </div>
                    <div className="flex flex-col gap-2">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => toggleException(word.id, word.is_exception)}
                      >
                        {word.is_exception ? '🚫 Bloquer' : '✅ Autoriser'}
                      </Button>
                      <Button
                        size="sm"
                        variant="destructive"
                        onClick={() => deleteWord(word.id, word.word)}
                      >
                        🗑️
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default BlockedWordsTab;
