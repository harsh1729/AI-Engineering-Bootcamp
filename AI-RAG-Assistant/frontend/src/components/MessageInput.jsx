import { useState } from "react";
import DocumentUpload from "./DocumentUpload";

export default function MessageInput({ onSend, disabled }) {
  const [value, setValue] = useState("");
  const [attachments, setAttachments] = useState([]); // [{ documentId, filename }]

  const handleSend = () => {
    const trimmed = value.trim();
    // Returning early here (no text) leaves `attachments` untouched, so any
    // already-uploaded documents stay attached for the next real send.
    if (!trimmed || disabled) return;

    onSend(trimmed, attachments.map((attachment) => attachment.documentId));
    setValue("");
    setAttachments([]);
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  const handleAttachmentUploaded = (attachment) => {
    setAttachments((prev) => [...prev, attachment]);
  };

  const handleRemoveAttachment = (documentId) => {
    setAttachments((prev) => prev.filter((attachment) => attachment.documentId !== documentId));
  };

  return (
    <div className="message-input">
      <div className="message-input-row">
        {/* Grouped so the chip list always matches the textarea's width,
            regardless of how much horizontal space Send/+ take up. */}
        <div className="message-input-main">
          {attachments.length > 0 && (
            <div className="attachment-list">
              {attachments.map((attachment) => (
                <div key={attachment.documentId} className="attachment-chip">
                  <span className="attachment-chip-icon" aria-hidden="true">
                    📄
                  </span>
                  <span className="attachment-chip-name">{attachment.filename}</span>
                  <button
                    type="button"
                    className="attachment-chip-remove"
                    onClick={() => handleRemoveAttachment(attachment.documentId)}
                    aria-label={`Remove ${attachment.filename}`}
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          )}

          <textarea
            value={value}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={handleKeyDown}
            rows={2}
            placeholder="Type your message..."
          />
        </div>

        <button type="button" onClick={handleSend} disabled={disabled}>
          {disabled ? "Thinking..." : "Send"}
        </button>

        <DocumentUpload attachmentCount={attachments.length} onUploaded={handleAttachmentUploaded} />
      </div>
    </div>
  );
}
