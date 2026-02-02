/**
 * AdminPanel.js
 * Panneau d'administration complet pour la gestion des licences, utilisateurs et mots bloqués
 */

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Badge } from './components/ui/badge';
import { toast } from 'sonner';

const AdminPanel = ({ onClose }) => {
  const [activeTab, setActiveTab] = useState('stats');
  const [loading, setLoading] = useState(false);
  
  // Stats
  const [stats, setStats] = useState(null);
  
  // Licences
  const [licenses, setLicenses] = useState([]);
  const [newLicense, setNewLicense] = useState({
    organization_name: '',
    license_key: '',
    max_users: 10,
    expiry_date: '',
    notes: ''
  });
  
  // Mots bloqués
  const [blockedWords, setBlockedWords] = useState([]);
  const [newWord, setNewWord] = useState({
    word: '',
    category: 'custom',
    severity: 'medium',
    is_exception: false
  });
  
  // Utilisateurs d'une licence
  const [selectedLicense, setSelectedLicense] = useState(null);
  const [licenseUsers, setLicenseUsers] = useState([]);
  const [showUsersModal, setShowUsersModal] = useState(false);
  const [selectedUsersForTransfer, setSelectedUsersForTransfer] = useState([]);

  const API = process.env.REACT_APP_BACKEND_URL;
  const token = localStorage.getItem('etienne_token');

  const axiosConfig = {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  };

  useEffect(() => {
    if (activeTab === 'stats') loadStats();
    if (activeTab === 'licenses') loadLicenses();
    if (activeTab === 'words') loadBlockedWords();
  }, [activeTab]);

  // ==================== STATISTIQUES ====================
  
  const loadStats = async () => {
    try {
      const response = await axios.get(`${API}/api/admin/stats`, axiosConfig);
      setStats(response.data.stats);
    } catch (error) {
      toast.error('Erreur chargement statistiques');
    }
  };

  // ==================== LICENCES ====================
  
  const loadLicenses = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/api/admin/licenses`, axiosConfig);
      setLicenses(response.data.licenses);
    } catch (error) {
      toast.error('Erreur chargement licences');
    } finally {
      setLoading(false);
    }
  };

  const createLicense = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      await axios.post(`${API}/api/admin/licenses`, newLicense, axiosConfig);
      toast.success('Licence créée avec succès');
      setNewLicense({
        organization_name: '',
        license_key: '',
        max_users: 10,
        expiry_date: '',
        notes: ''
      });
      loadLicenses();
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Erreur création licence');
    } finally {
      setLoading(false);
    }
  };

  const renewLicense = async (licenseKey, newExpiry) => {
    try {
      await axios.put(
        `${API}/api/admin/licenses/${licenseKey}`,
        { expiry_date: newExpiry },
        axiosConfig
      );
      toast.success('Licence renouvelée');
      loadLicenses();
    } catch (error) {
      toast.error('Erreur renouvellement');
    }
  };

  const toggleLicenseActive = async (licenseKey, currentStatus) => {
    try {
      await axios.put(
        `${API}/api/admin/licenses/${licenseKey}`,
        { is_active: !currentStatus },
        axiosConfig
      );
      toast.success(currentStatus ? 'Licence désactivée' : 'Licence activée');
      loadLicenses();
    } catch (error) {
      toast.error('Erreur modification');
    }
  };

  const updateMaxUsers = async (licenseKey, newMax) => {
    try {
      await axios.put(
        `${API}/api/admin/licenses/${licenseKey}`,
        { max_users: parseInt(newMax) },
        axiosConfig
      );
      toast.success('Limite modifiée');
      loadLicenses();
    } catch (error) {
      toast.error('Erreur modification');
    }
  };

  const setLicenseAdmin = async (licenseKey, adminEmail) => {
    try {
      await axios.post(
        `${API}/api/admin/licenses/${licenseKey}/set-admin`,
        { license_admin_email: adminEmail || null },
        axiosConfig
      );
      toast.success(adminEmail ? `Admin de licence désigné: ${adminEmail}` : 'Admin de licence retiré');
      loadLicenses();
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Erreur désignation admin');
    }
  };

  const updateLicenseKey = async (oldKey, newKey) => {
    try {
      // Utiliser l'endpoint de mise à jour existant avec un nouveau champ
      await axios.put(
        `${API}/api/admin/licenses/${oldKey}`,
        { license_key: newKey },
        axiosConfig
      );
      toast.success(`Clé modifiée: ${oldKey} → ${newKey}`);
      loadLicenses();
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Erreur modification clé');
    }
  };

  const viewLicenseUsers = async (licenseKey) => {
    try {
      const response = await axios.get(
        `${API}/api/admin/licenses/${licenseKey}/users`,
        axiosConfig
      );
      setLicenseUsers(response.data.users);
      setSelectedLicense(licenseKey);
      setShowUsersModal(true);
    } catch (error) {
      toast.error('Erreur chargement utilisateurs');
    }
  };

  // ==================== MOTS BLOQUÉS ====================
  
  const loadBlockedWords = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/api/admin/blocked-words`, axiosConfig);
      setBlockedWords(response.data.words);
    } catch (error) {
      toast.error('Erreur chargement mots bloqués');
    } finally {
      setLoading(false);
    }
  };

  const addBlockedWord = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      await axios.post(`${API}/api/admin/blocked-words`, newWord, axiosConfig);
      toast.success('Mot ajouté');
      setNewWord({ word: '', category: 'custom', severity: 'medium', is_exception: false });
      loadBlockedWords();
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Erreur ajout mot');
    } finally {
      setLoading(false);
    }
  };

  const deleteBlockedWord = async (wordId) => {
    try {
      await axios.delete(`${API}/api/admin/blocked-words/${wordId}`, axiosConfig);
      toast.success('Mot supprimé');
      loadBlockedWords();
    } catch (error) {
      toast.error('Erreur suppression');
    }
  };

  const toggleWordException = async (wordId, currentStatus) => {
    try {
      await axios.put(
        `${API}/api/admin/blocked-words/${wordId}`,
        { is_exception: !currentStatus },
        axiosConfig
      );
      toast.success('Statut modifié');
      loadBlockedWords();
    } catch (error) {
      toast.error('Erreur modification');
    }
  };

  return (
    <div className="admin-panel-overlay" onClick={onClose}>
      <div className="admin-panel-content" onClick={(e) => e.stopPropagation()} onMouseDown={(e) => e.stopPropagation()}>
        <Card className="w-full">
          <CardHeader>
            <div className="flex justify-between items-center">
              <CardTitle className="text-2xl font-bold">
                👨‍💼 Panneau d'Administration
              </CardTitle>
              <button 
                onClick={onClose}
                className="text-2xl hover:text-red-600 transition"
              >
                ✕
              </button>
            </div>
          </CardHeader>
          <CardContent>
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="stats" onClick={(e) => e.stopPropagation()}>📊 Stats</TabsTrigger>
                <TabsTrigger value="licenses" onClick={(e) => e.stopPropagation()}>📜 Licences</TabsTrigger>
                <TabsTrigger value="users" onClick={(e) => e.stopPropagation()}>👥 Utilisateurs</TabsTrigger>
                <TabsTrigger value="words" onClick={(e) => e.stopPropagation()}>🚫 Mots Bloqués</TabsTrigger>
              </TabsList>

              {/* ==================== ONGLET STATISTIQUES ==================== */}
              <TabsContent value="stats">
                <div className="space-y-6 mt-4">
                  <h3 className="text-xl font-semibold">📊 Statistiques Globales</h3>
                  {stats ? (
                    <>
                      {/* Cartes principales */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <Card className="bg-blue-50">
                          <CardContent className="p-4">
                            <div className="text-3xl font-bold text-blue-600">{stats.licenses?.total || 0}</div>
                            <div className="text-sm text-gray-600">Licences Totales</div>
                            <div className="text-xs text-green-600 mt-1">
                              {stats.licenses?.active || 0} actives • {stats.licenses?.expired || 0} expirées
                            </div>
                          </CardContent>
                        </Card>
                        <Card className="bg-green-50">
                          <CardContent className="p-4">
                            <div className="text-3xl font-bold text-green-600">{stats.users?.total || 0}</div>
                            <div className="text-sm text-gray-600">Utilisateurs Totaux</div>
                            <div className="text-xs text-green-600 mt-1">
                              {stats.users?.active || 0} actifs
                            </div>
                          </CardContent>
                        </Card>
                        <Card className="bg-purple-50">
                          <CardContent className="p-4">
                            <div className="text-3xl font-bold text-purple-600">{stats.logins?.total || 0}</div>
                            <div className="text-sm text-gray-600">Connexions Totales</div>
                            <div className="text-xs text-purple-600 mt-1">
                              Ce mois: {stats.logins?.this_month || 0}
                            </div>
                          </CardContent>
                        </Card>
                        <Card className="bg-orange-50">
                          <CardContent className="p-4">
                            <div className="text-3xl font-bold text-orange-600">{stats.requests?.total || 0}</div>
                            <div className="text-sm text-gray-600">Requêtes Totales</div>
                            <div className="text-xs text-orange-600 mt-1">
                              Ce mois: {stats.requests?.this_month || 0}
                            </div>
                          </CardContent>
                        </Card>
                      </div>
                      
                      {/* Statistiques par période */}
                      <div className="grid md:grid-cols-2 gap-6">
                        <Card>
                          <CardHeader>
                            <CardTitle className="text-lg">📅 Requêtes par Période</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="space-y-3">
                              <div className="flex justify-between items-center p-2 bg-blue-50 rounded">
                                <span>Ce mois ({stats.period?.current_month}/{stats.period?.current_year})</span>
                                <span className="font-bold text-blue-600">{stats.requests?.this_month || 0} requêtes</span>
                              </div>
                              <div className="flex justify-between items-center p-2 bg-green-50 rounded">
                                <span>Cette année ({stats.period?.current_year})</span>
                                <span className="font-bold text-green-600">{stats.requests?.this_year || 0} requêtes</span>
                              </div>
                              <div className="flex justify-between items-center p-2 bg-gray-50 rounded">
                                <span>Total historique</span>
                                <span className="font-bold text-gray-700">{stats.requests?.total || 0} requêtes</span>
                              </div>
                            </div>
                            
                            {/* Graphique par mois */}
                            {stats.requests?.by_month && (
                              <div className="mt-4">
                                <p className="text-sm font-medium mb-2">12 derniers mois:</p>
                                <div className="flex items-end gap-1 h-20">
                                  {stats.requests.by_month.slice().reverse().map((m, idx) => {
                                    const maxCount = Math.max(...stats.requests.by_month.map(x => x.count), 1);
                                    const height = (m.count / maxCount) * 100;
                                    return (
                                      <div key={idx} className="flex-1 flex flex-col items-center" title={`${m.month}/${m.year}: ${m.count}`}>
                                        <div 
                                          className="w-full bg-orange-400 rounded-t" 
                                          style={{ height: `${Math.max(height, 2)}%` }}
                                        ></div>
                                        <span className="text-[8px] text-gray-500">{m.month}</span>
                                      </div>
                                    );
                                  })}
                                </div>
                              </div>
                            )}
                          </CardContent>
                        </Card>
                        
                        <Card>
                          <CardHeader>
                            <CardTitle className="text-lg">🏆 Top Utilisateurs ce Mois</CardTitle>
                          </CardHeader>
                          <CardContent>
                            {stats.top_users?.this_month?.length > 0 ? (
                              <div className="space-y-2">
                                {stats.top_users.this_month.map((user, idx) => (
                                  <div key={idx} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                                    <span className="text-sm">
                                      {idx === 0 ? '🥇' : idx === 1 ? '🥈' : idx === 2 ? '🥉' : `${idx + 1}.`} {user.email}
                                    </span>
                                    <Badge>{user.count} req.</Badge>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <p className="text-gray-500 text-center">Aucune donnée ce mois</p>
                            )}
                          </CardContent>
                        </Card>
                      </div>
                      
                      {/* Tableau détaillé par utilisateur */}
                      <Card>
                        <CardHeader>
                          <CardTitle className="text-lg">👥 Statistiques Détaillées par Utilisateur</CardTitle>
                        </CardHeader>
                        <CardContent>
                          {stats.users_detailed?.length > 0 ? (
                            <div className="overflow-x-auto">
                              <table className="w-full text-sm">
                                <thead className="bg-gray-100">
                                  <tr>
                                    <th className="p-2 text-left">Email</th>
                                    <th className="p-2 text-right">Ce mois</th>
                                    <th className="p-2 text-right">Cette année</th>
                                    <th className="p-2 text-right">Total</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {stats.users_detailed.map((user, idx) => (
                                    <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                                      <td className="p-2">{user.email}</td>
                                      <td className="p-2 text-right font-medium text-blue-600">{user.this_month}</td>
                                      <td className="p-2 text-right font-medium text-green-600">{user.this_year}</td>
                                      <td className="p-2 text-right font-bold">{user.total}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          ) : (
                            <p className="text-gray-500 text-center">Aucune donnée disponible</p>
                          )}
                        </CardContent>
                      </Card>
                    </>
                  ) : (
                    <p className="text-center">⏳ Chargement des statistiques...</p>
                  )}
                </div>
              </TabsContent>

              {/* ==================== ONGLET LICENCES ==================== */}
              <TabsContent value="licenses">
                <div className="space-y-6 mt-4">
                  {/* Formulaire création licence */}
                  <Card className="bg-green-50">
                    <CardHeader>
                      <CardTitle className="text-lg">➕ Créer une Nouvelle Licence</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <form onSubmit={createLicense} className="space-y-3">
                        <div className="grid grid-cols-2 gap-3">
                          <Input
                            placeholder="Nom de l'organisation"
                            value={newLicense.organization_name}
                            onChange={(e) => setNewLicense({...newLicense, organization_name: e.target.value})}
                            required
                          />
                          <Input
                            placeholder="CLÉ-LICENCE-2024"
                            value={newLicense.license_key}
                            onChange={(e) => setNewLicense({...newLicense, license_key: e.target.value.toUpperCase()})}
                            required
                          />
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                          <Input
                            type="number"
                            placeholder="Nb max utilisateurs"
                            value={newLicense.max_users}
                            onChange={(e) => setNewLicense({...newLicense, max_users: parseInt(e.target.value)})}
                            required
                            min="1"
                          />
                          <Input
                            type="date"
                            value={newLicense.expiry_date}
                            onChange={(e) => setNewLicense({...newLicense, expiry_date: e.target.value})}
                            required
                          />
                        </div>
                        <Input
                          placeholder="Notes (optionnel)"
                          value={newLicense.notes}
                          onChange={(e) => setNewLicense({...newLicense, notes: e.target.value})}
                        />
                        <Button type="submit" disabled={loading} className="w-full">
                          {loading ? 'Création...' : 'Créer la Licence'}
                        </Button>
                      </form>
                    </CardContent>
                  </Card>

                  {/* Liste des licences */}
                  <div className="space-y-3">
                    <h3 className="text-lg font-semibold">Liste des Licences ({licenses.length})</h3>
                    {licenses.map(license => (
                      <Card key={license.id} className={license.is_expired ? 'border-red-300' : ''}>
                        <CardContent className="p-4">
                          <div className="flex justify-between items-start">
                            <div className="flex-1">
                              <div className="flex items-center gap-2">
                                <h4 className="font-bold text-lg">{license.organization_name}</h4>
                                {license.is_active ? (
                                  <Badge className="bg-green-500">Actif</Badge>
                                ) : (
                                  <Badge className="bg-red-500">Inactif</Badge>
                                )}
                                {license.is_expired && <Badge className="bg-orange-500">Expiré</Badge>}
                              </div>
                              <p className="text-sm text-gray-600 font-mono mt-1">{license.license_key}</p>
                              <div className="text-sm mt-2">
                                <span className="font-semibold">Utilisateurs:</span> {license.current_users}/{license.max_users} •
                                <span className="font-semibold ml-2">Expire:</span> {new Date(license.expiry_date).toLocaleDateString('fr-CA')}
                              </div>
                              {license.license_admin_email && (
                                <div className="text-sm mt-1 text-purple-600">
                                  <span className="font-semibold">👑 Admin de licence:</span> {license.license_admin_email}
                                </div>
                              )}
                              {license.notes && <p className="text-xs text-gray-500 mt-1">{license.notes}</p>}
                            </div>
                            <div className="flex flex-col gap-2">
                              <Button 
                                size="sm"
                                onClick={() => viewLicenseUsers(license.license_key)}
                              >
                                👥 Voir Utilisateurs
                              </Button>
                              <Button 
                                size="sm"
                                variant="outline"
                                className="text-purple-600 border-purple-300 hover:bg-purple-50"
                                onClick={() => {
                                  const adminEmail = prompt(
                                    `Désigner un admin de licence pour "${license.organization_name}":\n\n` +
                                    `Admin actuel: ${license.license_admin_email || 'Aucun'}\n\n` +
                                    `Entrez l'email d'un utilisateur de cette licence\n(ou laissez vide pour retirer l'admin):`,
                                    license.license_admin_email || ''
                                  );
                                  if (adminEmail !== null) {
                                    setLicenseAdmin(license.license_key, adminEmail);
                                  }
                                }}
                              >
                                👑 {license.license_admin_email ? 'Changer' : 'Désigner'} Admin
                              </Button>
                              <Button 
                                size="sm"
                                variant="outline"
                                onClick={() => {
                                  const newExpiry = prompt('Nouvelle date (YYYY-MM-DD):', license.expiry_date);
                                  if (newExpiry) renewLicense(license.license_key, newExpiry);
                                }}
                              >
                                🔄 Renouveler
                              </Button>
                              <Button 
                                size="sm"
                                variant="outline"
                                onClick={() => {
                                  const newMax = prompt('Nouvelle limite utilisateurs:', license.max_users);
                                  if (newMax) updateMaxUsers(license.license_key, newMax);
                                }}
                              >
                                ✏️ Modifier Limite
                              </Button>
                              <Button 
                                size="sm"
                                variant="outline"
                                onClick={() => {
                                  const newKey = prompt(
                                    `Modifier la clé de licence:\n\nClé actuelle: ${license.license_key}\n\nEntrez la nouvelle clé:`,
                                    license.license_key
                                  );
                                  if (newKey && newKey !== license.license_key) {
                                    updateLicenseKey(license.license_key, newKey.toUpperCase());
                                  }
                                }}
                              >
                                🔑 Modifier Clé
                              </Button>
                              <Button 
                                size="sm"
                                variant={license.is_active ? "destructive" : "default"}
                                onClick={() => toggleLicenseActive(license.license_key, license.is_active)}
                              >
                                {license.is_active ? '🔴 Désactiver' : '🟢 Activer'}
                              </Button>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </div>
              </TabsContent>

              {/* ==================== ONGLET UTILISATEURS ==================== */}
              <TabsContent value="users">
                <div className="space-y-4 mt-4">
                  {selectedLicense ? (
                    <>
                      <div className="flex justify-between items-center">
                        <h3 className="text-lg font-semibold">
                          Utilisateurs de la licence: {selectedLicense}
                        </h3>
                        <Button onClick={() => setSelectedLicense(null)} variant="outline">
                          ← Retour
                        </Button>
                      </div>
                      <div className="space-y-2">
                        {licenseUsers.map(user => (
                          <Card key={user.id}>
                            <CardContent className="p-3">
                              <div className="flex justify-between items-center">
                                <div>
                                  <p className="font-semibold">{user.full_name}</p>
                                  <p className="text-sm text-gray-600">{user.email}</p>
                                  <p className="text-xs text-gray-500">
                                    Inscrit: {new Date(user.created_at).toLocaleDateString('fr-CA')} •
                                    Messages: {user.message_count || 0}
                                  </p>
                                </div>
                                <Badge className={user.is_active ? 'bg-green-500' : 'bg-red-500'}>
                                  {user.is_active ? 'Actif' : 'Inactif'}
                                </Badge>
                              </div>
                            </CardContent>
                          </Card>
                        ))}
                        {licenseUsers.length === 0 && (
                          <p className="text-center text-gray-500 py-8">Aucun utilisateur pour cette licence</p>
                        )}
                      </div>
                    </>
                  ) : (
                    <div className="text-center py-12">
                      <p className="text-gray-500">Sélectionnez une licence dans l'onglet "Licences" pour voir ses utilisateurs</p>
                    </div>
                  )}
                </div>
              </TabsContent>

              {/* ==================== ONGLET MOTS BLOQUÉS ==================== */}
              <TabsContent value="words">
                <div className="space-y-6 mt-4">
                  {/* Formulaire ajout mot */}
                  <Card className="bg-red-50">
                    <CardHeader>
                      <CardTitle className="text-lg">➕ Ajouter un Mot/Phrase Bloqué(e)</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <form onSubmit={addBlockedWord} className="space-y-3">
                        <Input
                          placeholder="Mot ou phrase à bloquer"
                          value={newWord.word}
                          onChange={(e) => setNewWord({...newWord, word: e.target.value})}
                          required
                        />
                        <div className="grid grid-cols-2 gap-3">
                          <select
                            className="border rounded px-3 py-2"
                            value={newWord.category}
                            onChange={(e) => setNewWord({...newWord, category: e.target.value})}
                          >
                            <option value="violence">Violence</option>
                            <option value="drugs">Drogues</option>
                            <option value="illegal">Activités illégales</option>
                            <option value="hacking">Piratage</option>
                            <option value="inappropriate">Contenu inapproprié</option>
                            <option value="custom">Personnalisé</option>
                          </select>
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
                        </div>
                        <label className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={newWord.is_exception}
                            onChange={(e) => setNewWord({...newWord, is_exception: e.target.checked})}
                          />
                          <span className="text-sm">Exception (autoriser ce mot malgré détection)</span>
                        </label>
                        <Button type="submit" disabled={loading} className="w-full">
                          {loading ? 'Ajout...' : 'Ajouter le Mot'}
                        </Button>
                      </form>
                    </CardContent>
                  </Card>

                  {/* Liste des mots bloqués */}
                  <div className="space-y-3">
                    <h3 className="text-lg font-semibold">Mots/Phrases Bloqués ({blockedWords.length})</h3>
                    <div className="space-y-2 max-h-96 overflow-y-auto">
                      {blockedWords.map(word => (
                        <Card key={word.id} className={word.is_exception ? 'border-green-300' : 'border-red-300'}>
                          <CardContent className="p-3">
                            <div className="flex justify-between items-center">
                              <div className="flex-1">
                                <span className="font-mono font-semibold">{word.word}</span>
                                <div className="flex gap-2 mt-1">
                                  <Badge className="text-xs">{word.category}</Badge>
                                  <Badge className="text-xs" variant={
                                    word.severity === 'critical' ? 'destructive' :
                                    word.severity === 'high' ? 'default' : 'outline'
                                  }>
                                    {word.severity}
                                  </Badge>
                                  {word.is_exception && <Badge className="bg-green-500 text-xs">Exception</Badge>}
                                </div>
                              </div>
                              <div className="flex gap-2">
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => toggleWordException(word.id, word.is_exception)}
                                >
                                  {word.is_exception ? '🔒 Bloquer' : '✅ Autoriser'}
                                </Button>
                                <Button
                                  size="sm"
                                  variant="destructive"
                                  onClick={() => {
                                    if (window.confirm(`Supprimer "${word.word}" ?`)) {
                                      deleteBlockedWord(word.id);
                                    }
                                  }}
                                >
                                  🗑️
                                </Button>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>

      {/* Modal Utilisateurs d'une Licence */}
      {showUsersModal && (
        <UsersTransferModal
          selectedLicense={selectedLicense}
          licenseUsers={licenseUsers}
          licenses={licenses}
          onClose={() => {
            setShowUsersModal(false);
            setSelectedUsersForTransfer([]);
          }}
          onTransferComplete={() => {
            loadLicenses();
            viewLicenseUsers(selectedLicense);
          }}
          API={API}
          axiosConfig={axiosConfig}
        />
      )}
    </div>
  );
};

// Composant Modal pour la gestion et le transfert des utilisateurs
const UsersTransferModal = ({ selectedLicense, licenseUsers, licenses, onClose, onTransferComplete, API, axiosConfig }) => {
  const [selectedUsers, setSelectedUsers] = useState([]);
  const [targetLicense, setTargetLicense] = useState('');
  const [transferring, setTransferring] = useState(false);
  const [editingEmail, setEditingEmail] = useState(null);
  const [newEmail, setNewEmail] = useState('');
  const [changingEmail, setChangingEmail] = useState(false);
  const [editingPassword, setEditingPassword] = useState(null);
  const [newUserPassword, setNewUserPassword] = useState('');
  const [changingPassword, setChangingPassword] = useState(false);
  
  const generatePassword = () => {
    const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789!@#$%';
    let pwd = '';
    for (let i = 0; i < 10; i++) pwd += chars.charAt(Math.floor(Math.random() * chars.length));
    setNewUserPassword(pwd);
  };
  
  const changeUserPassword = async (userEmail) => {
    if (!newUserPassword || newUserPassword.length < 6) {
      toast.error('Le mot de passe doit contenir au moins 6 caractères');
      return;
    }
    setChangingPassword(true);
    try {
      await axios.post(`${API}/api/admin/change-user-password`, 
        { user_email: userEmail, new_password: newUserPassword },
        axiosConfig
      );
      toast.success(`Mot de passe modifié pour ${userEmail}`);
      setEditingPassword(null);
      setNewUserPassword('');
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Erreur');
    } finally {
      setChangingPassword(false);
    }
  };
  
  const toggleUserSelection = (email) => {
    setSelectedUsers(prev => 
      prev.includes(email) 
        ? prev.filter(e => e !== email)
        : [...prev, email]
    );
  };
  
  const selectAllUsers = () => {
    if (selectedUsers.length === licenseUsers.length) {
      setSelectedUsers([]);
    } else {
      setSelectedUsers(licenseUsers.map(u => u.email));
    }
  };
  
  const transferUsers = async () => {
    if (selectedUsers.length === 0) {
      toast.error('Sélectionnez au moins un utilisateur');
      return;
    }
    if (!targetLicense) {
      toast.error('Sélectionnez une licence cible');
      return;
    }
    if (targetLicense === selectedLicense) {
      toast.error('La licence cible doit être différente de la licence actuelle');
      return;
    }
    
    setTransferring(true);
    try {
      const response = await axios.post(
        `${API}/api/admin/transfer-users`,
        {
          user_emails: selectedUsers,
          target_license_key: targetLicense
        },
        axiosConfig
      );
      
      toast.success(response.data.message);
      
      if (response.data.errors && response.data.errors.length > 0) {
        response.data.errors.forEach(err => toast.error(err));
      }
      
      setSelectedUsers([]);
      setTargetLicense('');
      onTransferComplete();
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Erreur lors du transfert');
    } finally {
      setTransferring(false);
    }
  };
  
  const startEditEmail = (email) => {
    setEditingEmail(email);
    setNewEmail(email);
  };
  
  const cancelEditEmail = () => {
    setEditingEmail(null);
    setNewEmail('');
  };
  
  const changeUserEmail = async (oldEmail) => {
    if (!newEmail || newEmail === oldEmail) {
      toast.error('Veuillez entrer un nouvel email différent');
      return;
    }
    
    setChangingEmail(true);
    try {
      const response = await axios.post(
        `${API}/api/admin/change-user-email`,
        {
          old_email: oldEmail,
          new_email: newEmail
        },
        axiosConfig
      );
      
      toast.success(response.data.message);
      setEditingEmail(null);
      setNewEmail('');
      onTransferComplete();
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Erreur lors du changement d\'email');
    } finally {
      setChangingEmail(false);
    }
  };
  
  // Filtrer pour ne pas afficher la licence actuelle comme cible
  const availableLicenses = licenses.filter(l => l.license_key !== selectedLicense && l.is_active);
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <Card className="w-full max-w-4xl max-h-[85vh] overflow-hidden">
        <CardHeader>
          <div className="flex justify-between items-center">
            <CardTitle>
              👥 Utilisateurs de la licence: {selectedLicense}
            </CardTitle>
            <button
              onClick={onClose}
              className="text-2xl hover:text-red-600 transition"
            >
              ✕
            </button>
          </div>
        </CardHeader>
        <CardContent className="overflow-y-auto max-h-[65vh]">
          {licenseUsers.length === 0 ? (
            <div className="text-center p-8 text-gray-500">
              Aucun utilisateur pour cette licence
            </div>
          ) : (
            <>
              {/* Section de transfert */}
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                <h4 className="font-bold text-blue-800 mb-3">🔄 Transférer des utilisateurs</h4>
                
                <div className="flex flex-wrap gap-3 items-end">
                  <div className="flex-1 min-w-[200px]">
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Licence cible
                    </label>
                    <select
                      value={targetLicense}
                      onChange={(e) => setTargetLicense(e.target.value)}
                      className="w-full p-2 border rounded-md bg-white"
                    >
                      <option value="">-- Sélectionner une licence --</option>
                      {availableLicenses.map(lic => (
                        <option key={lic.license_key} value={lic.license_key}>
                          {lic.organization_name} ({lic.license_key}) - {lic.user_count || 0}/{lic.max_users} utilisateurs
                        </option>
                      ))}
                    </select>
                  </div>
                  
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      onClick={selectAllUsers}
                      size="sm"
                    >
                      {selectedUsers.length === licenseUsers.length ? '☐ Désélectionner tout' : '☑ Tout sélectionner'}
                    </Button>
                    
                    <Button
                      onClick={transferUsers}
                      disabled={transferring || selectedUsers.length === 0 || !targetLicense}
                      className="bg-blue-600 hover:bg-blue-700"
                    >
                      {transferring ? '⏳ Transfert...' : `🔄 Transférer (${selectedUsers.length})`}
                    </Button>
                  </div>
                </div>
                
                {selectedUsers.length > 0 && (
                  <div className="mt-2 text-sm text-blue-700">
                    {selectedUsers.length} utilisateur(s) sélectionné(s)
                  </div>
                )}
              </div>
              
              {/* Liste des utilisateurs */}
              <div className="space-y-3">
                {licenseUsers.map(user => (
                  <Card 
                    key={user.email} 
                    className={`cursor-pointer transition ${selectedUsers.includes(user.email) ? 'border-2 border-blue-500 bg-blue-50' : 'hover:bg-gray-50'}`}
                    onClick={() => toggleUserSelection(user.email)}
                  >
                    <CardContent className="p-4">
                      <div className="flex justify-between items-start">
                        <div className="flex items-start gap-3">
                          <div className="mt-1">
                            <input
                              type="checkbox"
                              checked={selectedUsers.includes(user.email)}
                              onChange={() => toggleUserSelection(user.email)}
                              onClick={(e) => e.stopPropagation()}
                              className="w-5 h-5 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                            />
                          </div>
                          <div>
                            <div className="flex items-center gap-2 mb-2">
                              <h4 className="font-bold">{user.full_name}</h4>
                              {user.is_active ? (
                                <Badge className="bg-green-500">Actif</Badge>
                              ) : (
                                <Badge className="bg-gray-500">Inactif</Badge>
                              )}
                              {user.role === 'license_admin' && (
                                <Badge className="bg-purple-500">Admin Licence</Badge>
                              )}
                            </div>
                            <div className="text-sm space-y-1 text-gray-600">
                              <div className="flex items-center gap-2">
                                📧 
                                {editingEmail === user.email ? (
                                  <div className="flex items-center gap-2 flex-1">
                                    <input
                                      type="email"
                                      value={newEmail}
                                      onChange={(e) => setNewEmail(e.target.value)}
                                      onClick={(e) => e.stopPropagation()}
                                      className="flex-1 p-1 border rounded text-sm"
                                      placeholder="Nouvel email"
                                    />
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        changeUserEmail(user.email);
                                      }}
                                      disabled={changingEmail}
                                      className="px-2 py-1 bg-green-500 text-white rounded text-xs hover:bg-green-600"
                                    >
                                      {changingEmail ? '...' : '✓'}
                                    </button>
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        cancelEditEmail();
                                      }}
                                      className="px-2 py-1 bg-gray-500 text-white rounded text-xs hover:bg-gray-600"
                                    >
                                      ✕
                                    </button>
                                  </div>
                                ) : (
                                  <>
                                    {user.email}
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        startEditEmail(user.email);
                                      }}
                                      className="ml-2 px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs hover:bg-blue-200"
                                      title="Modifier l'email"
                                    >
                                      ✏️ Modifier
                                    </button>
                                  </>
                                )}
                              </div>
                              <div>📅 Inscrit le: {new Date(user.created_at).toLocaleDateString('fr-CA')}</div>
                              <div>💬 Messages: {user.message_count || 0}</div>
                              {user.previous_email && (
                                <div className="text-purple-600">
                                  📧 Ancien email: {user.previous_email}
                                </div>
                              )}
                              {user.transferred_from && (
                                <div className="text-orange-600">
                                  ↩️ Transféré de: {user.transferred_from}
                                </div>
                              )}
                              {/* Bouton modifier mot de passe */}
                              <div className="mt-2">
                                {editingPassword === user.email ? (
                                  <div className="flex items-center gap-2 p-2 bg-yellow-50 rounded">
                                    <input
                                      type="text"
                                      value={newUserPassword}
                                      onChange={(e) => setNewUserPassword(e.target.value)}
                                      onClick={(e) => e.stopPropagation()}
                                      className="flex-1 p-1 border rounded text-sm"
                                      placeholder="Nouveau mot de passe"
                                    />
                                    <button onClick={(e) => { e.stopPropagation(); generatePassword(); }} className="px-2 py-1 bg-gray-200 rounded text-xs" title="Générer">🎲</button>
                                    <button onClick={(e) => { e.stopPropagation(); changeUserPassword(user.email); }} disabled={changingPassword} className="px-2 py-1 bg-green-500 text-white rounded text-xs">{changingPassword ? '...' : '✓'}</button>
                                    <button onClick={(e) => { e.stopPropagation(); setEditingPassword(null); setNewUserPassword(''); }} className="px-2 py-1 bg-gray-500 text-white rounded text-xs">✕</button>
                                  </div>
                                ) : (
                                  <button
                                    onClick={(e) => { e.stopPropagation(); setEditingPassword(user.email); }}
                                    className="px-2 py-1 bg-yellow-100 text-yellow-700 rounded text-xs hover:bg-yellow-200"
                                  >
                                    🔐 Modifier mot de passe
                                  </button>
                                )}
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default AdminPanel;
