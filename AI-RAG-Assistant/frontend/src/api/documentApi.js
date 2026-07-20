const UPLOAD_ENDPOINT = "http://127.0.0.1:8000/documents/upload";

async function readErrorMessage(response) {
  const data = await response.json().catch(() => null);
  return data?.detail || "Upload failed. Please try again.";
}

/**
 * Uploads a single file to the backend as multipart/form-data and returns the
 * parsed { document_id, filename, status } payload. Do not set a Content-Type
 * header manually - the browser generates the multipart boundary itself when
 * given a FormData body.
 */
export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);

  const controller = new AbortController();
  // Prevent the + button from spinning forever if the backend is down/hung.
  const timeoutId = setTimeout(() => controller.abort(), 30_000);

  let response;

  try {
    response = await fetch(UPLOAD_ENDPOINT, {
      method: "POST",
      body: formData,
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
