// Maximum number of documents a user can attach to a message before sending.
// Enforced client-side only - raising this is enough to change the limit
// everywhere it's read (DocumentUpload, MessageInput).
export const MAX_ATTACHMENTS = 3;

// Must stay in sync with backend app.config.MAX_DOCUMENT_SIZE_MB.
// Backend remains the source of truth; this is a UX guard only.
export const MAX_DOCUMENT_SIZE_MB = 15;
export const MAX_DOCUMENT_SIZE_BYTES = MAX_DOCUMENT_SIZE_MB * 1024 * 1024;
