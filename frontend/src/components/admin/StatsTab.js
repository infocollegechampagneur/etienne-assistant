/**
 * StatsTab.js
 * Onglet Statistiques du panneau admin
 */

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Card, CardContent } from '../ui/card';
import { toast } from 'sonner';

const StatsTab = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);

  const API = process.env.REACT_APP_BACKEND_URL;
  const token = localStorage.getItem('etienne_token');

  const axiosConfig = {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  };

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/api/admin/stats`, axiosConfig);
      setStats(response.data.stats);
    } catch (error) {
      toast.error('Érreur chargement statistiques');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="p-4 text-center">Chargement...</div>;
  }

  if (!stats) {
    return <div className="p-4 text-center text-gray-500">Aucune donnée disponible</div>;
  }

  return (
    <div className="space-y-4 mt-4">
      <h3 className="text-xl font-semibold">📊 Statistiques Globales</h3>
      <div className="grid grid-cols-2 gap-4">
        {/* Licences */}
        <Card>
          <CardContent className="p-4">
            <div className="text-3xl font-bold text-blue-600">{stats.licenses.total}</div>
            <div className="text-sm text-gray-600">Licences Totales</div>
            <div className="text-xs text-green-600 mt-1">
              {stats.licenses.active} actives • {stats.licenses.expired} expirées
            </div>
          </CardContent>
        </Card>

        {/* Utilisateurs */}
        <Card>
          <CardContent className="p-4">
            <div className="text-3xl font-bold text-green-600">{stats.users.total}</div>
            <div className="text-sm text-gray-600">Utilisateurs Totaux</div>
            <div className="text-xs text-green-600 mt-1">
              {stats.users.active} actifs
            </div>
          </CardContent>
        </Card>

        {/* Mots Bloqués */}
        <Card>
          <CardContent className="p-4">
            <div className="text-3xl font-bold text-orange-600">{stats.blocked_words}</div>
            <div className="text-sm text-gray-600">Mots Bloqués</div>
          </CardContent>
        </Card>

        {/* Messages Traités */}
        <Card>
          <CardContent className="p-4">
            <div className="text-3xl font-bold text-purple-600">{stats.messages_processed}</div>
            <div className="text-sm text-gray-600">Messages Traités</div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default StatsTab;
