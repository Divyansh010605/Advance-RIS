import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';

export default function LoginPage() {
  const { user, login, signup } = useAuth();
  const navigate = useNavigate();

  const [mode, setMode] = useState('login'); // 'login' | 'signup'
  const [form, setForm] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
  });
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [successMsg, setSuccessMsg] = useState('');

  useEffect(() => {
    if (user) navigate('/upload');
  }, [user, navigate]);

  function switchMode(newMode) {
    setMode(newMode);
    setError('');
    setSuccessMsg('');
    setShowPassword(false);
    setForm({ username: '', email: '', password: '', confirmPassword: '' });
  }

  function onChange(e) {
    setForm((f) => ({ ...f, [e.target.name]: e.target.value }));
    setError('');
    setSuccessMsg('');
  }

  function handleQuickAdmin() {
    switchMode('login');
    setForm({
      username: '',
      email: 'admin',
      password: 'admin123',
      confirmPassword: '',
    });
    setSuccessMsg('Admin credentials loaded! Click "Sign In" below.');
  }

  async function onSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccessMsg('');

    try {
      if (mode === 'login') {
        const identifier = form.email || form.username;
        if (!identifier || !identifier.trim()) {
          setError('Please enter your Username or Email Address.');
          setLoading(false);
          return;
        }
        if (!form.password) {
          setError('Please enter your Password.');
          setLoading(false);
          return;
        }
        await login(identifier.trim(), form.password);
      } else {
        // Signup validations
        if (!form.username || form.username.trim().length < 3) {
          setError('Username must be at least 3 characters long.');
          setLoading(false);
          return;
        }
        if (!form.email || !form.email.includes('@')) {
          setError('Please provide a valid email address.');
          setLoading(false);
          return;
        }
        if (!form.password || form.password.length < 8) {
          setError('Password must be at least 8 characters long.');
          setLoading(false);
          return;
        }
        if (form.password !== form.confirmPassword) {
          setError('Passwords do not match. Please check again.');
          setLoading(false);
          return;
        }

        await signup(form.username.trim(), form.email.trim(), form.password);
      }
      navigate('/upload');
    } catch (err) {
      setError(
        err.response?.data?.error ||
        err.message ||
        'Authentication failed. Please check your credentials and try again.'
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="auth-page" role="main">
      <div className="auth-card card" id="auth-card">
        {/* Header */}
        <header className="auth-header">
          <div className="auth-brand-logo">
            <span className="logo-accent">Lung</span>Detect AI
          </div>
          <h1>{mode === 'login' ? 'Welcome Back' : 'Create an Account'}</h1>
          <p>{mode === 'login' ? 'Sign in to access your screening history and analysis' : 'Join LungDetect AI for multi-disease lung screening'}</p>
        </header>

        {/* Quick Admin Access Preset Card */}
        <div className="admin-preset-card">
          <div className="admin-preset-header">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
              <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
            </svg>
            <strong>Default Admin Credentials</strong>
          </div>
          <div className="admin-preset-details">
            <span><strong>Username:</strong> <code>admin</code></span>
            <span><strong>Password:</strong> <code>admin123</code></span>
            <span><strong>Email:</strong> <code>admin01@example.com</code></span>
          </div>
          <button
            type="button"
            className="btn btn-preset-fill"
            onClick={handleQuickAdmin}
            id="quick-admin-btn"
          >
            ⚡ Use Admin Credentials
          </button>
        </div>

        {/* Mode Selector Tabs */}
        <div className="auth-tabs" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'login'}
            className={`auth-tab ${mode === 'login' ? 'active' : ''}`}
            onClick={() => switchMode('login')}
            id="tab-login"
          >
            Sign In
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'signup'}
            className={`auth-tab ${mode === 'signup' ? 'active' : ''}`}
            onClick={() => switchMode('signup')}
            id="tab-signup"
          >
            Create Account
          </button>
        </div>

        {/* Alert Error / Success Notifications */}
        {error && (
          <div className="alert alert-error mb-4" role="alert" id="auth-error">
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="8" x2="12" y2="12"></line>
                <line x1="12" y1="16" x2="12.01" y2="16"></line>
              </svg>
              <span>{error}</span>
            </div>
          </div>
        )}

        {successMsg && (
          <div className="alert alert-success mb-4" role="alert" id="auth-success">
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                <polyline points="22 4 12 14.01 9 11.01"></polyline>
              </svg>
              <span>{successMsg}</span>
            </div>
          </div>
        )}

        {/* Form */}
        <form onSubmit={onSubmit} id="auth-form" noValidate>
          {mode === 'signup' && (
            <div className="form-group">
              <label className="form-label" htmlFor="username">Username</label>
              <input
                id="username"
                name="username"
                type="text"
                className="form-input"
                placeholder="johndoe"
                value={form.username}
                onChange={onChange}
                required
                minLength={3}
                maxLength={30}
                autoComplete="username"
              />
              <span className="form-hint">3 to 30 characters</span>
            </div>
          )}

          <div className="form-group">
            <label className="form-label" htmlFor="email">
              {mode === 'login' ? 'Username or Email Address' : 'Email Address'}
            </label>
            <input
              id="email"
              name="email"
              type="text"
              className="form-input"
              placeholder={mode === 'login' ? 'admin or admin01@example.com' : 'you@hospital.com'}
              value={form.email}
              onChange={onChange}
              required
              autoComplete={mode === 'login' ? 'username' : 'email'}
            />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="password">Password</label>
            <div className="input-wrapper">
              <input
                id="password"
                name="password"
                type={showPassword ? 'text' : 'password'}
                className="form-input"
                placeholder="••••••••"
                value={form.password}
                onChange={onChange}
                required
                minLength={8}
                autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              />
              <button
                type="button"
                className="password-toggle-btn"
                onClick={() => setShowPassword(!showPassword)}
                aria-label={showPassword ? 'Hide password' : 'Show password'}
                tabIndex="-1"
              >
                {showPassword ? (
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path>
                    <line x1="1" y1="1" x2="23" y2="23"></line>
                  </svg>
                ) : (
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                    <circle cx="12" cy="12" r="3"></circle>
                  </svg>
                )}
              </button>
            </div>
          </div>

          {mode === 'signup' && (
            <div className="form-group">
              <label className="form-label" htmlFor="confirmPassword">Confirm Password</label>
              <input
                id="confirmPassword"
                name="confirmPassword"
                type={showPassword ? 'text' : 'password'}
                className="form-input"
                placeholder="••••••••"
                value={form.confirmPassword}
                onChange={onChange}
                required
                minLength={8}
                autoComplete="new-password"
              />
            </div>
          )}

          <button
            id="auth-submit"
            type="submit"
            className="btn btn-primary btn-lg"
            style={{ width: '100%', justifyContent: 'center', marginTop: '12px' }}
            disabled={loading}
          >
            {loading ? (
              <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span className="spinner-inline" /> Processing…
              </span>
            ) : mode === 'login' ? (
              'Sign In'
            ) : (
              'Create Account'
            )}
          </button>
        </form>

        <div className="auth-toggle">
          {mode === 'login' ? (
            <p>
              Don't have an account?{' '}
              <button id="auth-toggle-signup" onClick={() => switchMode('signup')}>
                Create one now
              </button>
            </p>
          ) : (
            <p>
              Already registered?{' '}
              <button id="auth-toggle-login" onClick={() => switchMode('login')}>
                Sign in here
              </button>
            </p>
          )}
        </div>

        <div className="divider" />
        <p className="text-center text-sm text-muted">
          Note: For research and educational use only. Not a certified medical diagnostic tool.
        </p>
      </div>
    </main>
  );
}
