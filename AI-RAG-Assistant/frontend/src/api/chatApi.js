import { getGuestId } from "../utils/guestId";

const CHAT_STREAM_ENDPOINT = "http://127.0.0.1:8000/chat/stream";

async function readErrorMessage(response) {
  const data = await response.json().catch(() => null);
  return data?.detail || "The assistant could not respond. Please try again.";
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

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunkText = decoder.decode(value, { stream: true });
    if (chunkText) {
      onChunk(chunkText);
    }
  }
}
