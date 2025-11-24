/**
 * UsersTab.js
 * Onglet Utilisateurs du panneau admin - NOUVELLE FONCTIONNALITÉ
 */

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Card, CardContent } from '../ui/card';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Badge } from '../ui/badge';
import { toast } from 'sonner';
import UserConversationsModal from './UserConversationsModal';

const UsersTab = () => {
  const [users, setUsers] = useState([]);
  const [filteredUsers, setFilteredUsers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterLicense, setFilterLicense] = useState('all');
  const [licenses, setLicenses] = useState([]);
  const [selectedUser, setSelectedUser] = useState(null);
  const [showConversationsModal, setShowConversationsModal] = useState(false);

  const API = process.env.REACT_APP_BACKEND_URL;
  const token = localStorage.getItem('etienne_token');

  const axiosConfig = {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  };

  useEffect(() => {
    loadUsers();
    loadLicenses();
  }, []);

  useEffect(() => {
    filterUsers();
  }, [users, searchTerm, filterLicense]);

  const loadUsers = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/api/admin/users`, axiosConfig);
      setUsers(response.data.users);
    } catch (error) {
      toast.error('Érreur chargement utilisateurs');
    } finally {
      setLoading(false);
    }
  };

  const loadLicenses = async () => {
    try {
      const response = await axios.get(`${API}/api/admin/licenses`, axiosConfig);
      setLicenses(response.data.licenses);
    } catch (error) {
      console.error('Erreur chargement licences:', error);
    }
  };

  const filterUsers = () => {
    let filtered = users;

    // Filtre par recherche
    if (searchTerm) {
      filtered = filtered.filter(user => 
        user.full_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        user.email.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Filtre par licence
    if (filterLicense !== 'all') {
      filtered = filtered.filter(user => user.license_key === filterLicense);
    }

    setFilteredUsers(filtered);
  };

  const toggleUserStatus = async (userId, currentStatus) => {
    try {
      await axios.put(
        `${API}/api/admin/users/${userId}`,
        { is_active: !currentStatus },
        axiosConfig
      );
      toast.success('Statut modifié');
      loadUsers();
    } catch (error) {
      toast.error('Érreur modification');
    }
  };

  const deleteUser = async (userId, userName) => {
    if (!window.confirm(`Supprimer l'utilisateur "${userName}" ? Cette action est irréversible.`)) {
      return;
    }

    try {
      await axios.delete(`${API}/api/admin/users/${userId}`, axiosConfig);
      toast.success('Utilisateur supprimé');
      loadUsers();
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Érreur suppression');
    }
  };

  const viewConversations = (user) => {
    setSelectedUser(user);
    setShowConversationsModal(true);
  };

  return (
    <div className="space-y-4 mt-4">
      {/* Filtres */}
      <div className="flex gap-3">
        <Input
          placeholder="Rechercher par nom ou email..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="flex-1"
        />
        <select
          className="border rounded px-3 py-2"
          value={filterLicense}
          onChange={(e) => setFilterLicense(e.target.value)}
        >
          <option value="all">Toutes les licences</option>
          {licenses.map(lic => (
            <option key={lic.license_key} value={lic.license_key}>
              {lic.organization_name}
            </option>
          ))}
        </select>
      </div>

      {/* Liste des utilisateurs */}
      <div className="space-y-3">
        <h3 className="text-lg font-semibold">
          Liste des Utilisateurs ({filteredUsers.length}/{users.length})
        </h3>
        
        {loading ? (
          <div className="text-center p-4">Chargement...</div>
        ) : filteredUsers.length === 0 ? (
          <div className="text-center p-4 text-gray-500">Aucun utilisateur trouvé</div>
        ) : (
          filteredUsers.map(user => (
            <Card key={user.id}>
              <CardContent className="p-4">
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <h4 className="font-bold">{user.full_name}</h4>
                      {user.is_active ? (
                        <Badge className="bg-green-500">Actif</Badge>
                      ) : (
                        <Badge className="bg-gray-500">Inactif</Badge>
                      )}
                      {user.is_admin && <Badge className="bg-orange-500">Admin</Badge>}
                    </div>
                    <div className="text-sm space-y-1">
                      <div>📧 {user.email}</div>
                      <div>🏢 {user.organization}</div>
                      <div>💬 {user.conversation_count || 0} conversations</div>
                      <div>📅 Inscrit le: {new Date(user.created_at).toLocaleDateString('fr-CA')}</div>
                    </div>
                  </div>
                  <div className="flex flex-col gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => viewConversations(user)}
                    >
                      💬 Historique
                    </Button>
                    <Button
                      size="sm"
                      variant={user.is_active ? "outline" : "default"}
                      onClick={() => toggleUserStatus(user.id, user.is_active)}
                    >
                      {user.is_active ? '⏸️ Désactiver' : '▶️ Activer'}
                    </Button>
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={() => deleteUser(user.id, user.full_name)}
                    >
                      🗑️ Supprimer
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>

      {/* Modal des conversations */}
      {showConversationsModal && selectedUser && (
        <UserConversationsModal
          user={selectedUser}
          onClose={() => {
            setShowConversationsModal(false);
            setSelectedUser(null);
          }}
        />
      )}
    </div>
  );
};

export default UsersTab;
