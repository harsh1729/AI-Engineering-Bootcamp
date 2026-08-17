import { API_BASE_URL, fetchJson } from "./apiClient";

export async function fetchAdminUsers() {
  return fetchJson(`${API_BASE_URL}/admin/users`);
}

export async function approveUser(userId) {
  return fetchJson(`${API_BASE_URL}/admin/users/${encodeURIComponent(userId)}/approve`, {
    method: "POST",
  });
}

export async function disableUser(userId) {
  return fetchJson(`${API_BASE_URL}/admin/users/${encodeURIComponent(userId)}/disable`, {
    method: "POST",
  });
}

export async function enableUser(userId) {
  return fetchJson(`${API_BASE_URL}/admin/users/${encodeURIComponent(userId)}/enable`, {
    method: "POST",
  });
}
