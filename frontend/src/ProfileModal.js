import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { toast } from 'sonner';

const ProfileModal = ({ onClose, API, onEmailChanged }) => {
  const [activeTab, setActiveTab] = useState('email');
  const [newEmail, setNewEmail] = useState('');
  const [password, setPassword] = useState('');
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentEmail, setCurrentEmail] = useState('');
  
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      try {
        const payload = JSON.parse(atob(token.split('.')[1]));
        setCurrentEmail(payload.email || '');
      } catch (e) {
        console.error('Erreur décodage token:', e);
      }
    }
  }, []);
  
  const handleChangeEmail = async (e) => {
    e.preventDefault();
    if (!newEmail || !password) {
      toast.error('Veuillez remplir tous les champs');
      return;
    }
    if (newEmail === currentEmail) {
      toast.error('Le nouvel email doit être différent');
      return;
    }
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await axios.post(
        `${API}/api/auth/change-email`,
        { new_email: newEmail, password: password },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      toast.success(response.data.message);
      onEmailChanged(response.data.new_token);
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Erreur lors du changement');
    } finally {
      setLoading(false);
    }
  };
  
  const handleChangePassword = async (e) => {
    e.preventDefault();
    if (!currentPassword || !newPassword || !confirmPassword) {
      toast.error('Veuillez remplir tous les champs');
      return;
    }
    if (newPassword !== confirmPassword) {
      toast.error('Les mots de passe ne correspondent pas');
      return;
    }
    if (newPassword.length < 6) {
      toast.error('Le mot de passe doit contenir au moins 6 caractères');
      return;
    }
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await axios.post(
        `${API}/api/auth/change-password`,
        { current_password: currentPassword, new_password: newPassword },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      toast.success(response.data.message);
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Erreur lors du changement');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <Card className="w-full max-w-md">
        <CardHeader>
          <div className="flex justify-between items-center">
            <CardTitle className="flex items-center gap-2">Mon Profil</CardTitle>
            <button onClick={onClose} className="text-2xl hover:text-red-600" data-testid="close-profile-modal">x</button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="mb-4 p-3 bg-gray-50 rounded-lg">
            <p className="text-sm text-gray-600">Email actuel:</p>
            <p className="font-semibold">{currentEmail}</p>
          </div>
          
          <div className="flex border-b mb-4">
            <button
              onClick={() => setActiveTab('email')}
              className={`flex-1 py-2 text-sm font-medium ${activeTab === 'email' ? 'border-b-2 border-blue-600 text-blue-600' : 'text-gray-500'}`}
            >
              Changer email
            </button>
            <button
              onClick={() => setActiveTab('password')}
              className={`flex-1 py-2 text-sm font-medium ${activeTab === 'password' ? 'border-b-2 border-blue-600 text-blue-600' : 'text-gray-500'}`}
            >
              Changer mot de passe
            </button>
          </div>
          
          {activeTab === 'email' ? (
            <form onSubmit={handleChangeEmail} className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Nouvel email</label>
                <Input type="email" value={newEmail} onChange={(e) => setNewEmail(e.target.value)} placeholder="nouveau@email.com" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Mot de passe actuel</label>
                <Input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Confirmez votre identité" />
              </div>
              <Button type="submit" className="w-full" disabled={loading}>
                {loading ? 'Modification...' : 'Changer mon email'}
              </Button>
            </form>
          ) : (
            <form onSubmit={handleChangePassword} className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Mot de passe actuel</label>
                <Input type="password" value={currentPassword} onChange={(e) => setCurrentPassword(e.target.value)} placeholder="Votre mot de passe actuel" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Nouveau mot de passe</label>
                <Input type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} placeholder="Min. 6 caractères" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Confirmer le nouveau mot de passe</label>
                <Input type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} placeholder="Répétez le nouveau mot de passe" />
              </div>
              <Button type="submit" className="w-full" disabled={loading}>
                {loading ? 'Modification...' : 'Changer mon mot de passe'}
              </Button>
            </form>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default ProfileModal;
