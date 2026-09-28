import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { login as apiLogin, signup as apiSignup, getMe } from '../services/authService';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => {
    try {
      const saved = sessionStorage.getItem('tb_user');
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });

  const [loading, setLoading] = useState(true);

  // Validate existing session token on initial mount
  useEffect(() => {
    async function initAuth() {
      const token = sessionStorage.getItem('tb_token');
      if (token) {
        try {
          const res = await getMe();
          if (res && res.user) {
            setUser(res.user);
            sessionStorage.setItem('tb_user', JSON.stringify(res.user));
          } else {
            sessionStorage.removeItem('tb_token');
            sessionStorage.removeItem('tb_user');
            setUser(null);
          }
        } catch {
          sessionStorage.removeItem('tb_token');
          sessionStorage.removeItem('tb_user');
          setUser(null);
        }
      } else {
        sessionStorage.removeItem('tb_user');
        setUser(null);
      }
      setLoading(false);
    }
    initAuth();
  }, []);

  const login = useCallback(async (identifier, password) => {
    const data = await apiLogin(identifier, password);
    sessionStorage.setItem('tb_token', data.token);
    sessionStorage.setItem('tb_user', JSON.stringify(data.user));
    setUser(data.user);
    return data;
  }, []);

  const signup = useCallback(async (username, email, password) => {
    const data = await apiSignup(username, email, password);
    sessionStorage.setItem('tb_token', data.token);
    sessionStorage.setItem('tb_user', JSON.stringify(data.user));
    setUser(data.user);
    return data;
  }, []);

  const logout = useCallback(() => {
    sessionStorage.removeItem('tb_token');
    sessionStorage.removeItem('tb_user');
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider value={{ user, loading, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider');
  return ctx;
}
