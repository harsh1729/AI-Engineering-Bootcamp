import { getGuestId } from "../utils/guestId";
import { API_BASE_URL, buildAuthHeaders, fetchJson } from "./apiClient";

async function readErrorMessage(response) {
  const data = await response.json().catch(() => null);
  return data?.detail || "Upload failed. Please try again.";
}

function buildDocumentParams() {
  const params = new URLSearchParams();
  params.set("guest_id", getGuestId());
  return params;
}

/**
 * Uploads a single file to the backend as multipart/form-data and returns the
 * parsed { document_id, filename, status } payload. Do not set a Content-Type
 * header manually - the browser generates the multipart boundary itself when
 * given a FormData body.
 */
export async function uploadDocument(file, ragOptions = null, { useAuth = true } = {}) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("guest_id", getGuestId());
  if (ragOptions) {
    formData.append("rag_options", JSON.stringify(ragOptions));
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 120_000);

  let response;

  try {
    response = await fetch(`${API_BASE_URL}/documents/upload`, {
      method: "POST",
      body: formData,
      headers: useAuth ? buildAuthHeaders() : {},
      signal: controller.signal,
    });
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new Error("Upload timed out. Is the backend running?");
    }
    throw new Error("Could not reach the server. Is the backend running?");
  } finally {
    clearTimeout(timeoutId);
  }

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json();
}

export async function listDocuments({ useAuth = true } = {}) {
  const params = buildDocumentParams();
  return fetchJson(`${API_BASE_URL}/documents?${params.toString()}`, {
    skipAuth: !useAuth,
  });
}
