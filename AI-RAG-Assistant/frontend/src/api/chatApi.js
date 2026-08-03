import { getGuestId } from "../utils/guestId";

const CHAT_STREAM_ENDPOINT = "http://127.0.0.1:8000/chat/stream";
const CHAT_ENDPOINT = "http://127.0.0.1:8000/chat";

async function readErrorMessage(response) {
  const data = await response.json().catch(() => null);
  return data?.detail || "The assistant could not respond. Please try again.";
}

/**
 * Sends a non-streaming RAG chat request when documents are attached.
 * Returns the grounded answer plus source metadata from retrieval.
 */
export async function sendRagChatMessage({
  provider,
  model,
  messages,
  documentIds,
  ragOptions = null,
}) {
  let response;

  try {
    response = await fetch(CHAT_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        guest_id: getGuestId(),
        provider,
        model,
        messages,
        document_ids: documentIds,
        rag_options: ragOptions,
      }),
    });
  } catch {
    throw new Error("Could not reach the server. Is the backend running?");
  }

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json();
}

/**
 * Sends the complete conversation to the streaming backend endpoint and
 * reads the response as it arrives, calling `onChunk` with each decoded
 * text chunk so the caller can render the assistant's reply in real time.
 * Throws an Error with a user-friendly message on any failure (including
 * the demo usage limit being reached, surfaced as the backend's own detail).
 */
export async function sendChatMessage({ provider, model, messages, documentIds, onChunk }) {
  let response;

  try {
    response = await fetch(CHAT_STREAM_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        guest_id: getGuestId(),
        provider,
        model,
        messages,
        document_ids: documentIds ?? [],
      }),
    });
  } catch {
    throw new Error("Could not reach the server. Is the backend running?");
  }

  if (!response.ok || !response.body) {
    throw new Error(await readErrorMessage(response));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let receivedText = false;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunkText = decoder.decode(value, { stream: true });
    if (chunkText) {
      receivedText = true;
      onChunk(chunkText);
    }
  }

  if (!receivedText) {
    throw new Error(
      "The model returned an empty response. Try a different model (e.g. gpt-4.1-mini)."
    );
  }
}
