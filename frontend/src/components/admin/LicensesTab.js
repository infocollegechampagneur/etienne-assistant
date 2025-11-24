/**
 * LicensesTab.js
 * Onglet Licences du panneau admin
 */

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Badge } from '../ui/badge';
import { toast } from 'sonner';

const LicensesTab = () => {
  const [licenses, setLicenses] = useState([]);
  const [loading, setLoading] = useState(false);
  const [newLicense, setNewLicense] = useState({
    organization_name: '',
    license_key: '',
    max_users: 10,
    expiry_date: '',
    notes: ''
  });

  const API = process.env.REACT_APP_BACKEND_URL;
  const token = localStorage.getItem('etienne_token');

  const axiosConfig = {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  };

  useEffect(() => {
    loadLicenses();
  }, []);

  const loadLicenses = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/api/admin/licenses`, axiosConfig);
      setLicenses(response.data.licenses);
    } catch (error) {
      toast.error('Érreur chargement licences');
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
      toast.error(error.response?.data?.detail || 'Érreur création licence');
    } finally {
      setLoading(false);
    }
  };

  const toggleLicense = async (licenseKey, currentStatus) => {
    try {
      await axios.put(
        `${API}/api/admin/licenses/${licenseKey}`,
        { is_active: !currentStatus },
        axiosConfig
      );
      toast.success('Statut modifié');
      loadLicenses();
    } catch (error) {
      toast.error('Érreur modification');
    }
  };

  return (
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
        {loading ? (
          <div className="text-center p-4">Chargement...</div>
        ) : (
          licenses.map(license => (
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
                    <div className="mt-2 space-y-1 text-sm">
                      <div><strong>Clé:</strong> <code className="bg-gray-100 px-2 py-1 rounded">{license.license_key}</code></div>
                      <div><strong>Utilisateurs:</strong> {license.current_users}/{license.max_users}</div>
                      <div><strong>Expire:</strong> {license.expiry_date}</div>
                      {license.notes && <div><strong>Notes:</strong> {license.notes}</div>}
                    </div>
                  </div>
                  <div className="flex flex-col gap-2">
                    <Button
                      size="sm"
                      variant={license.is_active ? "destructive" : "default"}
                      onClick={() => toggleLicense(license.license_key, license.is_active)}
                    >
                      {license.is_active ? '❌ Désactiver' : '✅ Activer'}
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </div>
  );
};

export default LicensesTab;
