import { getGuestId } from "../utils/guestId";
import { API_BASE_URL, fetchJson } from "./apiClient";

function buildChatHistoryParams(provider) {
  const params = new URLSearchParams({ provider });
  params.set("guest_id", getGuestId());
  return params;
}

export async function fetchChats(provider, { useAuth = true } = {}) {
  const params = buildChatHistoryParams(provider);
  return fetchJson(`${API_BASE_URL}/chats?${params.toString()}`, {
    skipAuth: !useAuth,
  });
}

export async function fetchChatMessages(chatId, provider, { useAuth = true } = {}) {
  const params = buildChatHistoryParams(provider);
  return fetchJson(
    `${API_BASE_URL}/chats/${encodeURIComponent(chatId)}/messages?${params.toString()}`,
    { skipAuth: !useAuth },
  );
}

export function mapPersistedMessages(messages) {
  return messages.map((message) => ({
    role: message.role,
    content: message.content,
  }));
}
