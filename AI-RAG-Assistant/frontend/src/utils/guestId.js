const GUEST_ID_STORAGE_KEY = "ai-rag-assistant:guest-id";

/**
 * Returns this browser's guest UUID, generating and persisting one in
 * localStorage on first visit so it's reused for every future request.
 */
export function getGuestId() {
  let guestId = localStorage.getItem(GUEST_ID_STORAGE_KEY);

  if (!guestId) {
    guestId = crypto.randomUUID();
    localStorage.setItem(GUEST_ID_STORAGE_KEY, guestId);
  }

  return guestId;
}
