import { useState } from "react";
import DocumentUpload from "./DocumentUpload";

export default function MessageInput({
  onSend,
  disabled,
  loadingLabel = "Thinking...",
  ragOptions,
  onAttachmentsChange,
}) {
  const [value, setValue] = useState("");
  const [attachments, setAttachments] = useState([]);
  // [{ documentId, filename, indexedChunkCount?, ragOptions }]

  const canSend = value.trim().length > 0 && !disabled;

  const updateAttachments = (updater) => {
    setAttachments((prev) => {
      const next = typeof updater === "function" ? updater(prev) : updater;
      onAttachmentsChange?.(next.length > 0);
      return next;
    });
  };

  const handleSend = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;

    const attachmentRagOptions =
      attachments.length > 0 ? attachments[0].ragOptions : null;

    onSend(
      trimmed,
      attachments.map((attachment) => attachment.documentId),
      attachmentRagOptions,
    );
    setValue("");
    updateAttachments([]);
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (canSend) {
        handleSend();
      }
    }
  };

  const handleAttachmentUploaded = (attachment) => {
    updateAttachments((prev) => [...prev, attachment]);
  };

  const handleRemoveAttachment = (documentId) => {
    updateAttachments((prev) =>
      prev.filter((attachment) => attachment.documentId !== documentId),
    );
  };

  const multipleDocuments = attachments.length > 1;
  const successMessage = multipleDocuments
    ? "✅ Documents processed successfully."
    : "✅ Document processed successfully.";
  const promptMessage = multipleDocuments
    ? "What would you like to know about these documents?"
    : "What would you like to know about this document?";

  return (
    <div className="message-input">
      <div className="message-input-row">
        <div className="message-input-main">
          {attachments.length > 0 && (
            <div className="attachment-list">
              {attachments.map((attachment, index) => {
                const isLastAttachment = index === attachments.length - 1;

                return (
                  <div key={attachment.documentId} className="attachment-chip">
                    <span className="attachment-chip-icon" aria-hidden="true">
                      📄
                    </span>
                    <div className="attachment-chip-body">
                      <div className="attachment-chip-header">
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
                      {isLastAttachment && (
                        <>
                          <p className="attachment-chip-success">{successMessage}</p>
                          <p className="attachment-chip-prompt">{promptMessage}</p>
                        </>
                      )}
                    </div>
                  </div>
                );
              })}
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

        <button
          type="button"
          className="message-input-send"
          onClick={handleSend}
          disabled={!canSend}
        >
          {disabled ? loadingLabel : "Send"}
        </button>

        <DocumentUpload
          attachmentCount={attachments.length}
          ragOptions={ragOptions}
          onUploaded={handleAttachmentUploaded}
        />
      </div>
    </div>
  );
}
