import { getGuestId } from "../utils/guestId";
import { API_BASE_URL, buildAuthHeaders, parseJsonResponse } from "./apiClient";

const CHAT_STREAM_ENDPOINT = `${API_BASE_URL}/chat/stream`;
const CHAT_ENDPOINT = `${API_BASE_URL}/chat`;

function buildChatRequestBody({
  provider,
  model,
  messages,
  documentIds,
  ragOptions,
  chatId,
}) {
  const body = {
    provider,
    model,
    messages,
    document_ids: documentIds,
    chat_id: chatId,
  };

  body.guest_id = getGuestId();

  if (ragOptions) {
    body.rag_options = ragOptions;
  }

  return body;
}

function parseSourcesHeader(response) {
  const rawSources = response.headers.get("X-RAG-Sources");
  if (!rawSources) {
    return [];
  }

  try {
    return JSON.parse(rawSources);
  } catch {
    return [];
  }
}

/**
 * Sends a non-streaming RAG chat request when documents are attached.
 * Returns the grounded answer plus source metadata from retrieval.
 *
 * Prefer sendChatMessage for the chat UI, which streams all replies.
 */
export async function sendRagChatMessage({
  provider,
  model,
  messages,
  documentIds,
  ragOptions = null,
  chatId = null,
}) {
  let response;

  try {
    response = await fetch(CHAT_ENDPOINT, {
      method: "POST",
      headers: buildAuthHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(
        buildChatRequestBody({
          provider,
          model,
          messages,
          documentIds,
          ragOptions,
          chatId,
        }),
      ),
    });
  } catch {
    throw new Error("Could not reach the server. Is the backend running?");
  }

  return parseJsonResponse(response);
}

/**
 * Sends the conversation to the streaming backend endpoint and reads the
 * response as it arrives. Calls onChunk for each text chunk and onSources
 * when grounded source metadata is available (before streaming begins).
 */
export async function sendChatMessage({
  provider,
  model,
  messages,
  documentIds,
  ragOptions = null,
  chatId = null,
  onChunk,
  onSources,
}) {
  let response;

  try {
    response = await fetch(CHAT_STREAM_ENDPOINT, {
      method: "POST",
      headers: buildAuthHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(
        buildChatRequestBody({
          provider,
          model,
          messages,
          documentIds: documentIds ?? [],
          ragOptions,
          chatId,
        }),
      ),
    });
  } catch {
    throw new Error("Could not reach the server. Is the backend running?");
  }

  if (!response.ok || !response.body) {
    throw new Error(await readErrorMessage(response));
  }

  const resolvedChatId = response.headers.get("X-Chat-Id");
  const sources = parseSourcesHeader(response);
  if (sources.length > 0 && onSources) {
    onSources(sources);
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
      "The model returned an empty response. Try a different model (e.g. gpt-4.1-mini).",
    );
  }

  return { chatId: resolvedChatId, sources };
}

async function readErrorMessage(response) {
  const data = await response.json().catch(() => null);
  return data?.detail || "The assistant could not respond. Please try again.";
}
