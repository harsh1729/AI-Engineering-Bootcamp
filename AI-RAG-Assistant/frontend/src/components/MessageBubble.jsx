const ROLE_LABELS = {
  user: "User",
  assistant: "Assistant",
  error: "Assistant",
};

export default function MessageBubble({
  role,
  content,
  warning,
  sources = [],
  pending,
  pendingLabel = "Thinking...",
}) {
  const hasContent = content && content.trim().length > 0;
  const hasSources = sources.length > 0;

  return (
    <div className={`message message-${role}`}>
      <div className="message-label">{ROLE_LABELS[role]}:</div>

      {hasContent && <div className="message-content">{content}</div>}

      {!hasContent && pending && (
        <div className="message-content message-content-empty">{pendingLabel}</div>
      )}

      {!hasContent && !pending && warning && (
        <div className="message-content message-content-empty">No answer was generated.</div>
      )}

      {hasSources && (
        <div className="message-sources">
          <span className="message-sources-label">Sources:</span>
          <ul>
            {sources.map((source) => (
              <li key={source.document_id}>
                {sources.length > 1 ? `[${source.source_number}] ` : ""}
                {source.filename}
              </li>
            ))}
          </ul>
        </div>
      )}

      {warning && <div className="message-warning">⚠️ {warning}</div>}
    </div>
  );
}
