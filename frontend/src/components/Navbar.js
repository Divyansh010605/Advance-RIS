import React from 'react';
import { NavLink, Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';

export default function Navbar() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  function handleLogout() {
    logout();
    navigate('/login');
  }

  return (
    <nav className="navbar" role="navigation" aria-label="Main navigation">
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <Link to="/upload" className="navbar__logo" id="nav-logo">
          <span>Lung</span>Detect AI
        </Link>
        {user && (
          <div className="navbar__user-badge" aria-label="Current user info">
            <span className="navbar__username">@{user.username}</span>
            {user.role === 'admin' && (
              <span className="navbar__role-tag role-admin">ADMIN</span>
            )}
            {user.role === 'radiologist' && (
              <span className="navbar__role-tag role-radiologist">RADIOLOGIST</span>
            )}
          </div>
        )}
      </div>
      <div className="navbar__links">
        {user ? (
          <>
            <NavLink
              to="/upload"
              id="nav-upload"
              className={({ isActive }) => `btn btn-ghost${isActive ? ' btn-primary' : ''}`}
            >
              Upload
            </NavLink>
            <NavLink
              to="/history"
              id="nav-history"
              className={({ isActive }) => `btn btn-ghost${isActive ? ' btn-primary' : ''}`}
            >
              History
            </NavLink>
            <NavLink
              to="/analytics"
              id="nav-analytics"
              className={({ isActive }) => `btn btn-ghost${isActive ? ' btn-primary' : ''}`}
            >
              Analytics
            </NavLink>

            <button id="nav-logout" className="btn btn-ghost" onClick={handleLogout}>
              Logout
            </button>
          </>
        ) : (
          <NavLink to="/login" id="nav-login" className="btn btn-primary">
            Sign In
          </NavLink>
        )}
      </div>
    </nav>
  );
}
