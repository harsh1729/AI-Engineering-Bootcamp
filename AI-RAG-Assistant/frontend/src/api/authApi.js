import { API_BASE_URL, fetchJson } from "./apiClient";
import { setAuthToken } from "../utils/authToken";

export async function registerUser({ name, email, password }) {
  return fetchJson(`${API_BASE_URL}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, email, password }),
  });
}

export async function loginUser({ email, password }) {
  const data = await fetchJson(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });

  setAuthToken(data.access_token);
  return data;
}

export async function fetchCurrentUser() {
  return fetchJson(`${API_BASE_URL}/auth/me`);
}
