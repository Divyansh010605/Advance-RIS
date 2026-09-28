import api from './api';

export async function login(identifier, password) {
  const res = await api.post('/auth/login', {
    email: identifier,
    username: identifier,
    identifier: identifier,
    password,
  });
  return res.data;
}

export async function signup(username, email, password) {
  const res = await api.post('/auth/signup', { username, email, password });
  return res.data;
}

export async function getMe() {
  const res = await api.get('/auth/me');
  return res.data;
}
