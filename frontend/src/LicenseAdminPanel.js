/**
 * LicenseAdminPanel.js
 * Panneau d'administration simplifié pour les admins de licence
 * Permet de: modifier la clé de licence, ajouter/retirer des utilisateurs
 */

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Badge } from './components/ui/badge';
import { toast } from 'sonner';

const API = process.env.REACT_APP_BACKEND_URL;

const LicenseAdminPanel = ({ onClose }) => {
  const [loading, setLoading] = useState(false);
  const [license, setLicense] = useState(null);
  const [users, setUsers] = useState([]);
  const [activeView, setActiveView] = useState('overview'); // 'overview', 'users', 'settings'
  
  // Formulaire nouvel utilisateur
  const [newUser, setNewUser] = useState({
    full_name: '',
    email: '',
    password: ''
  });
  
  // Formulaire modification clé
  const [newLicenseKey, setNewLicenseKey] = useState('');
  const [showKeyChange, setShowKeyChange] = useState(false);

  const token = localStorage.getItem('etienne_token');
  const axiosConfig = {
    headers: { 'Authorization': `Bearer ${token}` }
  };

  useEffect(() => {
    loadLicenseData();
    loadUsers();
  }, []);

  const loadLicenseData = async () => {
    try {
      const response = await axios.get(`${API}/api/license-admin/my-license`, axiosConfig);
      setLicense(response.data.license);
    } catch (error) {
      if (error.response?.status === 403) {
        toast.error("Vous n'êtes pas administrateur d'une licence");
        onClose();
      } else {
        toast.error('Erreur chargement licence');
      }
    }
  };

  const loadUsers = async () => {
    try {
      const response = await axios.get(`${API}/api/license-admin/users`, axiosConfig);
      setUsers(response.data.users);
    } catch (error) {
      console.error('Erreur chargement utilisateurs:', error);
    }
  };

  const addUser = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      await axios.post(`${API}/api/license-admin/users`, newUser, axiosConfig);
      toast.success(`Utilisateur ${newUser.email} ajouté avec succès`);
      setNewUser({ full_name: '', email: '', password: '' });
      loadUsers();
      loadLicenseData();
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Erreur ajout utilisateur');
    } finally {
      setLoading(false);
    }
  };

  const removeUser = async (userId, userEmail) => {
    if (!window.confirm(`Êtes-vous sûr de vouloir retirer ${userEmail} de la licence?`)) {
      return;
    }
    
    try {
      await axios.delete(`${API}/api/license-admin/users/${userId}`, axiosConfig);
      toast.success(`Utilisateur ${userEmail} retiré`);
      loadUsers();
      loadLicenseData();
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Erreur suppression utilisateur');
    }
  };

  const updateLicenseKey = async (e) => {
    e.preventDefault();
    
    if (!newLicenseKey.trim()) {
      toast.error('Veuillez entrer une nouvelle clé');
      return;
    }
    
    if (!window.confirm(`Modifier la clé de licence?\n\nAncienne: ${license?.license_key}\nNouvelle: ${newLicenseKey}\n\nTous les utilisateurs devront utiliser la nouvelle clé.`)) {
      return;
    }
    
    setLoading(true);
    
    try {
      await axios.put(`${API}/api/license-admin/update-key`, 
        { new_license_key: newLicenseKey.toUpperCase() }, 
        axiosConfig
      );
      toast.success('Clé de licence modifiée avec succès');
      setShowKeyChange(false);
      setNewLicenseKey('');
      loadLicenseData();
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Erreur modification clé');
    } finally {
      setLoading(false);
    }
  };

  const generatePassword = () => {
    const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789!@#$%';
    let password = '';
    for (let i = 0; i < 10; i++) {
      password += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    setNewUser({ ...newUser, password });
    toast.success('Mot de passe généré');
  };

  if (!license) {
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <Card className="w-96 p-8 text-center">
          <div className="animate-spin h-8 w-8 border-4 border-orange-500 border-t-transparent rounded-full mx-auto mb-4"></div>
          <p>Chargement...</p>
        </Card>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" onClick={onClose}>
      <Card className="w-full max-w-4xl max-h-[90vh] overflow-hidden" onClick={(e) => e.stopPropagation()}>
        <CardHeader className="bg-gradient-to-r from-blue-500 to-purple-600 text-white">
          <div className="flex justify-between items-center">
            <div>
              <CardTitle className="text-xl">🏢 Administration de Licence</CardTitle>
              <p className="text-blue-100 mt-1">{license.organization_name}</p>
            </div>
            <button onClick={onClose} className="text-2xl hover:text-red-300 transition">✕</button>
          </div>
        </CardHeader>
        
        <CardContent className="p-0">
          {/* Navigation */}
          <div className="flex border-b">
            <button 
              onClick={() => setActiveView('overview')}
              className={`flex-1 py-3 px-4 text-center transition ${activeView === 'overview' ? 'bg-blue-50 text-blue-600 border-b-2 border-blue-600' : 'hover:bg-gray-50'}`}
            >
              📊 Vue d'ensemble
            </button>
            <button 
              onClick={() => setActiveView('users')}
              className={`flex-1 py-3 px-4 text-center transition ${activeView === 'users' ? 'bg-blue-50 text-blue-600 border-b-2 border-blue-600' : 'hover:bg-gray-50'}`}
            >
              👥 Utilisateurs ({users.length}/{license.max_users})
            </button>
            <button 
              onClick={() => setActiveView('settings')}
              className={`flex-1 py-3 px-4 text-center transition ${activeView === 'settings' ? 'bg-blue-50 text-blue-600 border-b-2 border-blue-600' : 'hover:bg-gray-50'}`}
            >
              ⚙️ Paramètres
            </button>
          </div>

          <div className="p-6 max-h-[60vh] overflow-y-auto">
            {/* VUE D'ENSEMBLE */}
            {activeView === 'overview' && (
              <div className="space-y-6">
                <div className="grid grid-cols-2 gap-4">
                  <Card className="bg-blue-50">
                    <CardContent className="p-4">
                      <div className="text-3xl font-bold text-blue-600">{license.current_users}</div>
                      <div className="text-sm text-gray-600">Utilisateurs actifs</div>
                      <div className="text-xs text-gray-500 mt-1">sur {license.max_users} maximum</div>
                    </CardContent>
                  </Card>
                  
                  <Card className={license.is_expired ? 'bg-red-50' : 'bg-green-50'}>
                    <CardContent className="p-4">
                      <div className="text-lg font-bold text-gray-800">
                        {new Date(license.expiry_date).toLocaleDateString('fr-CA')}
                      </div>
                      <div className="text-sm text-gray-600">Date d'expiration</div>
                      {license.is_expired ? (
                        <Badge className="bg-red-500 mt-2">Expirée</Badge>
                      ) : (
                        <Badge className="bg-green-500 mt-2">Active</Badge>
                      )}
                    </CardContent>
                  </Card>
                </div>
                
                <Card>
                  <CardContent className="p-4">
                    <h3 className="font-semibold mb-2">🔑 Clé de licence actuelle</h3>
                    <code className="bg-gray-100 px-3 py-2 rounded block text-center font-mono">
                      {license.license_key}
                    </code>
                    <p className="text-xs text-gray-500 mt-2 text-center">
                      Partagez cette clé avec les nouveaux utilisateurs pour qu'ils puissent s'inscrire
                    </p>
                  </CardContent>
                </Card>
                
                <div className="text-sm text-gray-500 text-center">
                  <p>💡 En tant qu'admin de licence, vous pouvez:</p>
                  <p>• Ajouter/retirer des utilisateurs (dans la limite de {license.max_users})</p>
                  <p>• Modifier la clé de licence</p>
                  <p className="text-orange-600 mt-2">⚠️ Pour augmenter la limite ou prolonger la licence, contactez l'administrateur principal</p>
                </div>
              </div>
            )}

            {/* GESTION DES UTILISATEURS */}
            {activeView === 'users' && (
              <div className="space-y-6">
                {/* Formulaire ajout utilisateur */}
                <Card className="bg-green-50 border-green-200">
                  <CardHeader>
                    <CardTitle className="text-lg">➕ Ajouter un Utilisateur</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {license.current_users >= license.max_users ? (
                      <div className="text-center p-4 bg-red-50 rounded-lg">
                        <p className="text-red-600 font-semibold">
                          ⚠️ Limite d'utilisateurs atteinte ({license.max_users} max)
                        </p>
                        <p className="text-sm text-gray-600 mt-2">
                          Contactez le super administrateur pour augmenter la limite
                        </p>
                      </div>
                    ) : (
                      <form onSubmit={addUser} className="space-y-3">
                        <Input
                          placeholder="Nom complet"
                          value={newUser.full_name}
                          onChange={(e) => setNewUser({...newUser, full_name: e.target.value})}
                          required
                        />
                        <Input
                          type="email"
                          placeholder="Adresse email"
                          value={newUser.email}
                          onChange={(e) => setNewUser({...newUser, email: e.target.value})}
                          required
                        />
                        <div className="flex gap-2">
                          <Input
                            type="text"
                            placeholder="Mot de passe"
                            value={newUser.password}
                            onChange={(e) => setNewUser({...newUser, password: e.target.value})}
                            required
                            className="flex-1"
                          />
                          <Button type="button" variant="outline" onClick={generatePassword}>
                            🎲 Générer
                          </Button>
                        </div>
                        <Button type="submit" disabled={loading} className="w-full">
                          {loading ? 'Ajout...' : 'Ajouter l\'utilisateur'}
                        </Button>
                      </form>
                    )}
                  </CardContent>
                </Card>

                {/* Liste des utilisateurs */}
                <div className="space-y-2">
                  <h3 className="font-semibold">
                    👥 Utilisateurs ({users.length}/{license.max_users})
                  </h3>
                  {users.length === 0 ? (
                    <p className="text-center text-gray-500 py-8">Aucun utilisateur dans cette licence</p>
                  ) : (
                    users.map(user => (
                      <Card key={user.id} className="hover:shadow-md transition">
                        <CardContent className="p-4">
                          <div className="flex justify-between items-center">
                            <div>
                              <p className="font-semibold">{user.full_name}</p>
                              <p className="text-sm text-gray-600">{user.email}</p>
                              <p className="text-xs text-gray-500">
                                Inscrit: {new Date(user.created_at).toLocaleDateString('fr-CA')}
                              </p>
                            </div>
                            <div className="flex items-center gap-3">
                              <Badge className={user.is_active ? 'bg-green-500' : 'bg-gray-500'}>
                                {user.is_active ? 'Actif' : 'Inactif'}
                              </Badge>
                              <Button 
                                size="sm" 
                                variant="destructive"
                                onClick={() => removeUser(user.id, user.email)}
                              >
                                🗑️ Retirer
                              </Button>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))
                  )}
                </div>
              </div>
            )}

            {/* PARAMÈTRES */}
            {activeView === 'settings' && (
              <div className="space-y-6">
                {/* Modification de la clé */}
                <Card className="border-orange-200">
                  <CardHeader>
                    <CardTitle className="text-lg">🔑 Modifier la clé de licence</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-gray-600 mb-4">
                      Changer la clé de licence mettra à jour automatiquement tous les utilisateurs existants.
                      Les nouveaux utilisateurs devront utiliser la nouvelle clé pour s'inscrire.
                    </p>
                    
                    <div className="bg-gray-50 p-3 rounded-lg mb-4">
                      <p className="text-sm text-gray-500">Clé actuelle:</p>
                      <code className="font-mono font-bold">{license.license_key}</code>
                    </div>
                    
                    {showKeyChange ? (
                      <form onSubmit={updateLicenseKey} className="space-y-3">
                        <Input
                          placeholder="NOUVELLE-CLE-LICENCE-2025"
                          value={newLicenseKey}
                          onChange={(e) => setNewLicenseKey(e.target.value.toUpperCase())}
                          required
                          className="font-mono"
                        />
                        <div className="flex gap-2">
                          <Button type="submit" disabled={loading} className="flex-1">
                            {loading ? 'Modification...' : '✅ Confirmer'}
                          </Button>
                          <Button 
                            type="button" 
                            variant="outline" 
                            onClick={() => {
                              setShowKeyChange(false);
                              setNewLicenseKey('');
                            }}
                          >
                            Annuler
                          </Button>
                        </div>
                      </form>
                    ) : (
                      <Button 
                        onClick={() => setShowKeyChange(true)} 
                        variant="outline" 
                        className="w-full"
                      >
                        ✏️ Modifier la clé de licence
                      </Button>
                    )}
                  </CardContent>
                </Card>

                {/* Informations de licence (lecture seule) */}
                <Card className="bg-gray-50">
                  <CardHeader>
                    <CardTitle className="text-lg">ℹ️ Informations de la licence</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Organisation:</span>
                      <span className="font-semibold">{license.organization_name}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Limite utilisateurs:</span>
                      <span className="font-semibold">{license.max_users}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Date d'expiration:</span>
                      <span className="font-semibold">{new Date(license.expiry_date).toLocaleDateString('fr-CA')}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Statut:</span>
                      <Badge className={license.is_active && !license.is_expired ? 'bg-green-500' : 'bg-red-500'}>
                        {license.is_expired ? 'Expirée' : license.is_active ? 'Active' : 'Inactive'}
                      </Badge>
                    </div>
                    
                    <div className="mt-4 p-3 bg-yellow-50 rounded-lg text-sm">
                      <p className="text-yellow-800">
                        ⚠️ Pour modifier la limite d'utilisateurs ou prolonger la licence, 
                        contactez le super administrateur.
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default LicenseAdminPanel;
