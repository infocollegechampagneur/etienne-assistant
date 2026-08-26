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
  const [keysStatus, setKeysStatus] = useState(null);
  const [showKeys, setShowKeys] = useState(false);

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

  const fetchKeysStatus = async () => {
    try {
      const response = await axios.get(`${API}/api/keys-status`);
      setKeysStatus(response.data);
    } catch (error) {
      console.error('Erreur récupération statut clés:', error);
    }
  };

  useEffect(() => {
    fetchKeysStatus();
    const interval = setInterval(fetchKeysStatus, 15000);
    return () => clearInterval(interval);
  }, []);

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
            {quota.remaining} requête{quota.remaining > 1 ? 's' : ''} disponible{quota.remaining > 1 ? 's' : ''}
          </span>
        ) : quota.status === 'api_cooldown' ? (
          <span className="text-amber-600 font-medium">
            API en pause temporaire
          </span>
        ) : (
          <span className="text-red-500 font-medium">
            Limite atteinte
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
        <div className={`mt-2 p-2 ${quota.status === 'api_cooldown' ? 'bg-amber-50 border-amber-200 text-amber-700' : 'bg-orange-50 border-orange-200 text-orange-700'} border rounded text-xs`}>
          {countdown > 0
            ? (quota.status === 'api_cooldown' 
                ? <>L'API d'Étienne a besoin d'une pause. Réessai dans <strong>{formatTime(countdown)}</strong></>
                : <>Nouvelles requêtes dans <strong>{formatTime(countdown)}</strong></>)
            : <span className="text-green-600 font-medium">Vous pouvez réessayer maintenant</span>
          }
        </div>
      )}

      {/* Tableau de bord des clés API */}
      {keysStatus && keysStatus.total_keys > 0 && (
        <div className="mt-2 pt-2 border-t border-gray-200">
          <button
            data-testid="keys-dashboard-toggle"
            onClick={() => setShowKeys(prev => !prev)}
            className="w-full flex items-center justify-between text-xs text-gray-600 hover:text-gray-900 transition-colors duration-200"
          >
            <span className="font-medium">
              🔑 Clés API : {keysStatus.active_keys}/{keysStatus.total_keys} active{keysStatus.active_keys > 1 ? 's' : ''}
            </span>
            <span className="flex items-center gap-1">
              <span className={keysStatus.estimated_total_remaining > 10 ? 'text-green-600 font-semibold' : keysStatus.estimated_total_remaining > 0 ? 'text-orange-600 font-semibold' : 'text-red-600 font-semibold'}>
                ~{keysStatus.estimated_total_remaining} restantes aujourd'hui
              </span>
              <span className="text-gray-400">{showKeys ? '▲' : '▼'}</span>
            </span>
          </button>

          {showKeys && (
            <div data-testid="keys-dashboard-list" className="mt-2 space-y-1.5">
              {keysStatus.keys.map((k) => {
                const usagePct = Math.min(100, (k.usage_today / k.daily_limit_estimate) * 100);
                return (
                  <div key={k.index} data-testid={`key-status-row-${k.index}`} className="bg-gray-50 border border-gray-100 rounded p-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-mono text-gray-700">Clé {k.index} ({k.masked})</span>
                      {k.status === 'active' && <span className="text-green-600 font-medium">🟢 Active</span>}
                      {k.status === 'cooldown' && <span className="text-amber-600 font-medium">🟠 Pause {k.cooldown_remaining}s</span>}
                      {k.status === 'exhausted' && <span className="text-red-600 font-medium">🔴 Quota épuisé</span>}
                    </div>
                    <div className="flex items-center gap-2 mt-1">
                      <div className="flex-1 bg-gray-200 rounded-full h-1.5">
                        <div
                          className={`h-1.5 rounded-full transition-all duration-300 ${k.status === 'exhausted' ? 'bg-red-400' : usagePct > 75 ? 'bg-orange-400' : 'bg-green-400'}`}
                          style={{ width: `${usagePct}%` }}
                        />
                      </div>
                      <span className="text-[10px] text-gray-500 whitespace-nowrap">
                        {k.usage_today}/{k.daily_limit_estimate} auj.
                      </span>
                    </div>
                  </div>
                );
              })}
              <div className="text-[10px] text-gray-400 pt-0.5">
                Limite estimée : ~{keysStatus.daily_limit_per_key} requêtes/jour par projet Google. Ajoutez des clés de <strong>projets différents</strong> pour augmenter le total.
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default QuotaDisplay;
