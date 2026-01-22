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
      <div className="admin-panel-content" onClick={(e) => e.stopPropagation()}>
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
              <TabsList className="grid w-full grid-cols-4" onClick={(e) => e.stopPropagation()}>
                <TabsTrigger value="stats">📊 Stats</TabsTrigger>
                <TabsTrigger value="licenses">📜 Licences</TabsTrigger>
                <TabsTrigger value="users">👥 Utilisateurs</TabsTrigger>
                <TabsTrigger value="words">🚫 Mots Bloqués</TabsTrigger>
              </TabsList>

              {/* ==================== ONGLET STATISTIQUES ==================== */}
              <TabsContent value="stats">
                <div className="space-y-4 mt-4">
                  <h3 className="text-xl font-semibold">Statistiques Globales</h3>
                  {stats ? (
                    <div className="grid grid-cols-2 gap-4">
                      <Card>
                        <CardContent className="p-4">
                          <div className="text-3xl font-bold text-blue-600">{stats.licenses.total}</div>
                          <div className="text-sm text-gray-600">Licences Totales</div>
                          <div className="text-xs text-green-600 mt-1">
                            {stats.licenses.active} actives • {stats.licenses.expired} expirées
                          </div>
                        </CardContent>
                      </Card>
                      <Card>
                        <CardContent className="p-4">
                          <div className="text-3xl font-bold text-green-600">{stats.users.total}</div>
                          <div className="text-sm text-gray-600">Utilisateurs Totaux</div>
                          <div className="text-xs text-green-600 mt-1">
                            {stats.users.active} actifs
                          </div>
                        </CardContent>
                      </Card>
                      <Card>
                        <CardContent className="p-4">
                          <div className="text-3xl font-bold text-orange-600">{stats.blocked_words}</div>
                          <div className="text-sm text-gray-600">Mots Bloqués</div>
                        </CardContent>
                      </Card>
                      <Card>
                        <CardContent className="p-4">
                          <div className="text-3xl font-bold text-purple-600">{stats.messages_processed}</div>
                          <div className="text-sm text-gray-600">Messages Traités</div>
                        </CardContent>
                      </Card>
                    </div>
                  ) : (
                    <p>Chargement...</p>
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
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <Card className="w-full max-w-3xl max-h-[80vh] overflow-hidden">
            <CardHeader>
              <div className="flex justify-between items-center">
                <CardTitle>
                  👥 Utilisateurs de la licence {selectedLicense}
                </CardTitle>
                <button
                  onClick={() => setShowUsersModal(false)}
                  className="text-2xl hover:text-red-600 transition"
                >
                  ✕
                </button>
              </div>
            </CardHeader>
            <CardContent className="overflow-y-auto max-h-[60vh]">
              {licenseUsers.length === 0 ? (
                <div className="text-center p-8 text-gray-500">
                  Aucun utilisateur pour cette licence
                </div>
              ) : (
                <div className="space-y-3">
                  {licenseUsers.map(user => (
                    <Card key={user.id}>
                      <CardContent className="p-4">
                        <div className="flex justify-between items-start">
                          <div>
                            <div className="flex items-center gap-2 mb-2">
                              <h4 className="font-bold">{user.full_name}</h4>
                              {user.is_active ? (
                                <Badge className="bg-green-500">Actif</Badge>
                              ) : (
                                <Badge className="bg-gray-500">Inactif</Badge>
                              )}
                            </div>
                            <div className="text-sm space-y-1">
                              <div>📧 {user.email}</div>
                              <div>📅 Inscrit le: {new Date(user.created_at).toLocaleDateString('fr-CA')}</div>
                              <div>💬 Messages: {user.message_count || 0}</div>
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
};

export default AdminPanel;
