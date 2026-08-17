export const API_BASE_URL = "http://127.0.0.1:8000";

import { clearAuthToken, getAuthToken } from "../utils/authToken";

export function buildAuthHeaders(extraHeaders = {}) {
  const headers = { ...extraHeaders };
  const token = getAuthToken();
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  return headers;
}

async function readErrorMessage(response) {
  const data = await response.json().catch(() => null);
  return data?.detail || "Request failed. Please try again.";
}

export async function parseJsonResponse(response) {
  if (response.status === 401) {
    clearAuthToken();
  }

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json();
}

export async function fetchJson(url, options = {}) {
  const { skipAuth = false, headers: optionHeaders, ...fetchOptions } = options;
  const headers = skipAuth
    ? { ...(optionHeaders ?? {}) }
    : buildAuthHeaders({ ...(optionHeaders ?? {}) });
  let response;
  try {
    response = await fetch(url, {
      ...fetchOptions,
      headers,
    });
  } catch {
    throw new Error("Could not reach the server. Is the backend running?");
  }

  return parseJsonResponse(response);
}
