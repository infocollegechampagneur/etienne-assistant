/**
 * AuthModal.js
 * Modal de connexion/inscription pour le système de licences
 * Inclut la fonctionnalité de réinitialisation de mot de passe
 */

import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import axios from 'axios';

const AuthModal = ({ isOpen, onClose, onSuccess }) => {
  const [activeTab, setActiveTab] = useState('login');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  
  // Login form
  const [loginEmail, setLoginEmail] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  
  // Signup form
  const [signupName, setSignupName] = useState('');
  const [signupEmail, setSignupEmail] = useState('');
  const [signupPassword, setSignupPassword] = useState('');
  const [signupLicenseKey, setSignupLicenseKey] = useState('');
  
  // Password reset
  const [showForgotPassword, setShowForgotPassword] = useState(false);
  const [resetEmail, setResetEmail] = useState('');
  const [resetToken, setResetToken] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showResetForm, setShowResetForm] = useState(false);

  const API = process.env.REACT_APP_BACKEND_URL;

  // Vérifier si l'URL contient un token de réinitialisation
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const token = urlParams.get('reset_token');
    if (token) {
      setResetToken(token);
      setShowResetForm(true);
      setShowForgotPassword(true);
      // Nettoyer l'URL
      window.history.replaceState({}, document.title, window.location.pathname);
    }
  }, []);

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const response = await axios.post(`${API}/api/auth/login`, {
        email: loginEmail,
        password: loginPassword
      });

      if (response.data.success) {
        // Sauvegarder le token
        localStorage.setItem('etienne_token', response.data.token);
        localStorage.setItem('etienne_user', JSON.stringify(response.data.user));
        
        onSuccess(response.data.user);
        onClose();
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Erreur de connexion');
    } finally {
      setLoading(false);
    }
  };

  const handleForgotPassword = async (e) => {
    e.preventDefault();
    setError('');
    setSuccessMessage('');
    setLoading(true);

    try {
      const response = await axios.post(`${API}/api/auth/forgot-password`, {
        email: resetEmail
      });

      if (response.data.success) {
        setSuccessMessage('Si cette adresse email est associée à un compte, vous recevrez un lien de réinitialisation par email.');
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Erreur lors de la demande');
    } finally {
      setLoading(false);
    }
  };

  const handleResetPassword = async (e) => {
    e.preventDefault();
    setError('');
    setSuccessMessage('');

    if (newPassword !== confirmPassword) {
      setError('Les mots de passe ne correspondent pas');
      return;
    }

    if (newPassword.length < 6) {
      setError('Le mot de passe doit contenir au moins 6 caractères');
      return;
    }

    setLoading(true);

    try {
      const response = await axios.post(`${API}/api/auth/reset-password`, {
        token: resetToken,
        new_password: newPassword
      });

      if (response.data.success) {
        setSuccessMessage('Mot de passe réinitialisé avec succès! Vous pouvez maintenant vous connecter.');
        setShowResetForm(false);
        setShowForgotPassword(false);
        setResetToken('');
        setNewPassword('');
        setConfirmPassword('');
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Erreur lors de la réinitialisation');
    } finally {
      setLoading(false);
    }
  };

  const handleSignup = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const response = await axios.post(`${API}/api/auth/signup`, {
        full_name: signupName,
        email: signupEmail,
        password: signupPassword,
        license_key: signupLicenseKey
      });

      if (response.data.success) {
        // Sauvegarder le token
        localStorage.setItem('etienne_token', response.data.token);
        localStorage.setItem('etienne_user', JSON.stringify(response.data.user));
        
        onSuccess(response.data.user);
        onClose();
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Erreur lors de l\'inscription');
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  // Formulaire de réinitialisation de mot de passe
  if (showForgotPassword) {
    return (
      <div className="auth-modal-overlay" onClick={onClose}>
        <div className="auth-modal-content" onClick={(e) => e.stopPropagation()}>
          <Card className="w-full max-w-md">
            <CardHeader>
              <CardTitle className="text-center text-2xl font-bold">
                🔑 {showResetForm ? 'Nouveau mot de passe' : 'Mot de passe oublié'}
              </CardTitle>
              <p className="text-center text-sm text-gray-600 mt-2">
                {showResetForm 
                  ? 'Entrez votre nouveau mot de passe'
                  : 'Entrez votre email pour recevoir un lien de réinitialisation'}
              </p>
            </CardHeader>
            <CardContent>
              {showResetForm ? (
                // Formulaire de nouveau mot de passe
                <form onSubmit={handleResetPassword} className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Nouveau mot de passe</label>
                    <Input
                      type="password"
                      value={newPassword}
                      onChange={(e) => setNewPassword(e.target.value)}
                      placeholder="••••••••"
                      required
                      minLength={6}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Confirmer le mot de passe</label>
                    <Input
                      type="password"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      placeholder="••••••••"
                      required
                    />
                  </div>

                  {error && (
                    <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
                      {error}
                    </div>
                  )}

                  {successMessage && (
                    <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded">
                      {successMessage}
                    </div>
                  )}

                  <Button type="submit" className="w-full" disabled={loading}>
                    {loading ? 'Réinitialisation...' : 'Réinitialiser le mot de passe'}
                  </Button>
                </form>
              ) : (
                // Formulaire de demande de réinitialisation
                <form onSubmit={handleForgotPassword} className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Adresse email</label>
                    <Input
                      type="email"
                      value={resetEmail}
                      onChange={(e) => setResetEmail(e.target.value)}
                      placeholder="votre.email@ecole.qc.ca"
                      required
                    />
                  </div>

                  {error && (
                    <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
                      {error}
                    </div>
                  )}

                  {successMessage && (
                    <div className="bg-green-50 border-2 border-green-300 text-green-700 px-4 py-3 rounded-lg">
                      <p className="font-medium">✅ Email envoyé!</p>
                      <p className="text-sm mt-1">{successMessage}</p>
                    </div>
                  )}

                  <Button type="submit" className="w-full" disabled={loading || successMessage}>
                    {loading ? 'Envoi...' : successMessage ? '✓ Email envoyé' : 'Envoyer le lien de réinitialisation'}
                  </Button>
                </form>
              )}

              <div className="mt-4 text-center">
                <button
                  type="button"
                  onClick={() => {
                    setShowForgotPassword(false);
                    setShowResetForm(false);
                    setError('');
                    setSuccessMessage('');
                  }}
                  className="text-sm text-blue-600 hover:underline"
                >
                  ← Retour à la connexion
                </button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="auth-modal-overlay" onClick={onClose}>
      <div className="auth-modal-content" onClick={(e) => e.stopPropagation()}>
        <Card className="w-full max-w-md">
          <CardHeader>
            <CardTitle className="text-center text-2xl font-bold">
              🔐 Accès à Étienne
            </CardTitle>
            <p className="text-center text-sm text-gray-600 mt-2">
              Plateforme pour enseignants du secondaire québécois
            </p>
          </CardHeader>
          <CardContent>
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="login">Connexion</TabsTrigger>
                <TabsTrigger value="signup">Inscription</TabsTrigger>
              </TabsList>

              {/* Login Tab */}
              <TabsContent value="login">
                <form onSubmit={handleLogin} className="space-y-4 mt-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Email</label>
                    <Input
                      type="email"
                      value={loginEmail}
                      onChange={(e) => setLoginEmail(e.target.value)}
                      placeholder="votre.email@ecole.qc.ca"
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Mot de passe</label>
                    <Input
                      type="password"
                      value={loginPassword}
                      onChange={(e) => setLoginPassword(e.target.value)}
                      placeholder="••••••••"
                      required
                    />
                  </div>

                  {error && (
                    <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
                      {error}
                    </div>
                  )}

                  {successMessage && (
                    <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded">
                      {successMessage}
                    </div>
                  )}

                  <Button 
                    type="submit" 
                    className="w-full"
                    disabled={loading}
                  >
                    {loading ? 'Connexion...' : 'Se connecter'}
                  </Button>
                  
                  <div className="text-center mt-2">
                    <button
                      type="button"
                      onClick={() => {
                        setShowForgotPassword(true);
                        setError('');
                        setSuccessMessage('');
                      }}
                      className="text-sm text-blue-600 hover:underline"
                    >
                      Mot de passe oublié?
                    </button>
                  </div>
                </form>
              </TabsContent>

              {/* Signup Tab */}
              <TabsContent value="signup">
                <form onSubmit={handleSignup} className="space-y-4 mt-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Nom complet</label>
                    <Input
                      type="text"
                      value={signupName}
                      onChange={(e) => setSignupName(e.target.value)}
                      placeholder="Jean Tremblay"
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Email</label>
                    <Input
                      type="email"
                      value={signupEmail}
                      onChange={(e) => setSignupEmail(e.target.value)}
                      placeholder="votre.email@ecole.qc.ca"
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Mot de passe</label>
                    <Input
                      type="password"
                      value={signupPassword}
                      onChange={(e) => setSignupPassword(e.target.value)}
                      placeholder="••••••••"
                      required
                      minLength={6}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Clé de licence</label>
                    <Input
                      type="text"
                      value={signupLicenseKey}
                      onChange={(e) => setSignupLicenseKey(e.target.value.toUpperCase())}
                      placeholder="ETIENNE-ECOLE-2024"
                      required
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      La clé fournie par votre établissement
                    </p>
                  </div>

                  {error && (
                    <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
                      {error}
                    </div>
                  )}

                  <Button 
                    type="submit" 
                    className="w-full"
                    disabled={loading}
                  >
                    {loading ? 'Création du compte...' : 'Créer mon compte'}
                  </Button>
                </form>
              </TabsContent>
            </Tabs>

            <div className="mt-6 text-center">
              <button 
                onClick={onClose}
                className="text-sm text-gray-600 hover:text-gray-800"
              >
                Fermer
              </button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default AuthModal;
