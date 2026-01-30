/**
 * QuotaDisplay.js
 * Affiche le statut du quota de requêtes Gemini avec countdown
 */

import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API = process.env.REACT_APP_BACKEND_URL;

const QuotaDisplay = () => {
  const [quota, setQuota] = useState(null);
  const [countdown, setCountdown] = useState(0);

  // Récupérer le statut du quota
  const fetchQuota = async () => {
    try {
      const response = await axios.get(`${API}/api/quota-status`);
      setQuota(response.data);
      setCountdown(response.data.reset_in_seconds || 0);
    } catch (error) {
      console.error('Erreur récupération quota:', error);
    }
  };

  // Rafraîchir le quota toutes les 5 secondes
  useEffect(() => {
    fetchQuota();
    const interval = setInterval(fetchQuota, 5000);
    return () => clearInterval(interval);
  }, []);

  // Countdown local
  useEffect(() => {
    if (countdown > 0) {
      const timer = setTimeout(() => {
        setCountdown(prev => Math.max(0, prev - 1));
      }, 1000);
      return () => clearTimeout(timer);
    } else if (quota && !quota.can_request) {
      // Quand le countdown atteint 0, rafraîchir le quota
      fetchQuota();
    }
  }, [countdown, quota]);

  if (!quota) return null;

  const percentage = ((quota.max - quota.remaining) / quota.max) * 100;
  
  // Couleurs selon le statut
  let barColor = 'bg-green-500';
  let textColor = 'text-green-600';
  if (quota.remaining <= 3) {
    barColor = 'bg-red-500';
    textColor = 'text-red-600';
  } else if (quota.remaining <= 7) {
    barColor = 'bg-orange-500';
    textColor = 'text-orange-600';
  }

  const formatTime = (seconds) => {
    if (seconds <= 0) return '0s';
    if (seconds < 60) return `${seconds}s`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };

  return (
    <div className="quota-display bg-white/80 backdrop-blur-sm border border-gray-200 rounded-lg p-3 shadow-sm">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-700">
          🔋 Requêtes IA
        </span>
        <span className={`text-sm font-bold ${textColor}`}>
          {quota.remaining}/{quota.max}
        </span>
      </div>
      
      {/* Barre de progression */}
      <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
        <div 
          className={`h-2 rounded-full transition-all duration-300 ${barColor}`}
          style={{ width: `${100 - percentage}%` }}
        />
      </div>
      
      {/* Message de statut */}
      <div className="flex items-center justify-between text-xs">
        {quota.can_request ? (
          <span className="text-gray-500">
            ✅ {quota.remaining} requête{quota.remaining > 1 ? 's' : ''} disponible{quota.remaining > 1 ? 's' : ''}
          </span>
        ) : (
          <span className="text-red-500 font-medium">
            ⏳ Quota épuisé
          </span>
        )}
        
        {/* Countdown jusqu'au reset */}
        {countdown > 0 && (
          <span className={`font-medium ${quota.can_request ? 'text-gray-500' : 'text-orange-600'}`}>
            🔄 Reset: {formatTime(countdown)}
          </span>
        )}
      </div>
      
      {/* Message d'attente si quota épuisé */}
      {!quota.can_request && (
        <div className="mt-2 p-2 bg-orange-50 border border-orange-200 rounded text-xs text-orange-700">
          ⏰ Nouvelles requêtes dans <strong>{formatTime(countdown)}</strong>
        </div>
      )}
    </div>
  );
};

export default QuotaDisplay;
