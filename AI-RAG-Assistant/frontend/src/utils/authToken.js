const AUTH_TOKEN_STORAGE_KEY = "ai-rag-assistant:auth-token";

export function getAuthToken() {
  return localStorage.getItem(AUTH_TOKEN_STORAGE_KEY);
}

export function setAuthToken(token) {
  localStorage.setItem(AUTH_TOKEN_STORAGE_KEY, token);
}

export function clearAuthToken() {
  localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
}

export function isAuthenticated() {
  return Boolean(getAuthToken());
}
