import { useState } from "react";
import DocumentUpload from "./DocumentUpload";
import NewChatButton from "./NewChatButton";

function SendArrowIcon() {
  return (
    <svg
      className="message-input-send-icon"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <path
        d="M12 4L12 16M12 4L7 9M12 4L17 9"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function MessageInput({
  onSend,
  onStartNewChat,
  disabled,
  loadingLabel = "Thinking...",
  ragOptions,
  onAttachmentsChange,
}) {
  const [value, setValue] = useState("");
  const [attachments, setAttachments] = useState([]);

  const canSend = value.trim().length > 0 && !disabled;
  const isLoading = disabled && value.trim().length > 0;

  const updateAttachments = (updater) => {
    setAttachments((prev) => {
      const next = typeof updater === "function" ? updater(prev) : updater;
      onAttachmentsChange?.(next.length > 0, next);
      return next;
    });
  };

  const uploadRagOptions =
    attachments.length > 0 ? attachments[0].ragOptions ?? ragOptions : ragOptions;

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

  const getAttachmentFeedback = (items) => {
    const imageCount = items.filter((item) => item.contentType === "image").length;
    const documentCount = items.length - imageCount;

    if (imageCount === items.length) {
      return {
        successMessage:
          imageCount > 1
            ? "✅ Images processed successfully."
            : "✅ Image processed successfully.",
        promptMessage:
          imageCount > 1
            ? "Ask about text visible in these images. Only OCR-extracted text is searchable—not visual details."
            : "Ask about text visible in this image. Only OCR-extracted text is searchable—not visual details.",
      };
    }

    if (documentCount === items.length) {
      return {
        successMessage:
          documentCount > 1
            ? "✅ Documents processed successfully."
            : "✅ Document processed successfully.",
        promptMessage:
          documentCount > 1
            ? "What would you like to know about these documents?"
            : "What would you like to know about this document?",
      };
    }

    return {
      successMessage: "✅ Attachments processed successfully.",
      promptMessage:
        "Ask about document content or OCR text from images. Visual details in photos aren't searchable.",
    };
  };

  const { successMessage, promptMessage } = getAttachmentFeedback(attachments);

  const attachmentIcon = (contentType) => (contentType === "image" ? "🖼️" : "📄");

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
                      {attachmentIcon(attachment.contentType)}
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

        <div className="message-input-actions">
          <button
            type="button"
            className="message-input-send"
            onClick={handleSend}
            disabled={!canSend}
            aria-label={isLoading ? loadingLabel : "Send message"}
          >
            {isLoading ? (
              <span className="message-input-send-spinner" aria-hidden="true" />
            ) : (
              <SendArrowIcon />
            )}
          </button>

          <DocumentUpload
            attachmentCount={attachments.length}
            ragOptions={uploadRagOptions}
            onUploaded={handleAttachmentUploaded}
          />

          <NewChatButton onStartNewChat={onStartNewChat} disabled={disabled} />
        </div>
      </div>
    </div>
  );
}
