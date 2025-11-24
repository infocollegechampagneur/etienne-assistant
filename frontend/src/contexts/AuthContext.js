/**
 * AuthContext.js
 * Context global pour gérer l'authentification dans toute l'application
 */

import React, { createContext, useState, useContext, useEffect } from 'react';
import axios from 'axios';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [currentUser, setCurrentUser] = useState(null);
  const [isAdmin, setIsAdmin] = useState(false);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [loading, setLoading] = useState(true);

  const API = process.env.REACT_APP_BACKEND_URL;

  // Liste des emails administrateurs
  const adminEmails = ['informatique@champagneur.qc.ca'];

  // Vérifier l'authentification au chargement
  useEffect(() => {
    checkAuth();
  }, []);

  const checkAuth = () => {
    const storedUser = localStorage.getItem('etienne_user');
    const storedToken = localStorage.getItem('etienne_token');
    
    if (storedUser && storedToken) {
      try {
        const user = JSON.parse(storedUser);
        setCurrentUser(user);
        setIsAdmin(adminEmails.includes(user.email));
      } catch (error) {
        console.error('Erreur parsing user:', error);
        localStorage.removeItem('etienne_user');
        localStorage.removeItem('etienne_token');
      }
    }
    setLoading(false);
  };

  const login = (user, token) => {
    localStorage.setItem('etienne_token', token);
    localStorage.setItem('etienne_user', JSON.stringify(user));
    setCurrentUser(user);
    setIsAdmin(adminEmails.includes(user.email));
    setShowAuthModal(false);
  };

  const logout = () => {
    localStorage.removeItem('etienne_token');
    localStorage.removeItem('etienne_user');
    setCurrentUser(null);
    setIsAdmin(false);
  };

  const value = {
    currentUser,
    isAdmin,
    showAuthModal,
    setShowAuthModal,
    login,
    logout,
    loading,
    checkAuth
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};
