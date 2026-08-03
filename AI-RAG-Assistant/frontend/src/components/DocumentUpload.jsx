import { useEffect, useRef, useState } from "react";
import { uploadDocument } from "../api/documentApi";
import { MAX_ATTACHMENTS, MAX_DOCUMENT_SIZE_BYTES, MAX_DOCUMENT_SIZE_MB } from "../constants/attachments";
import spinnerIcon from "../assets/spinner.svg";

// Keep in sync with backend app.config.ALLOWED_DOCUMENT_EXTENSIONS
// (modern formats only - legacy .doc/.xls/.ppt are rejected server-side if enabled).
const ACCEPTED_FILE_TYPES = ".pdf,.txt,.docx,.xlsx,.pptx,.rtf";
const ERROR_MESSAGE_DURATION_MS = 6000;

function DocumentUpload({ attachmentCount, onUploaded, ragOptions }) {
  const containerRef = useRef(null);
  const fileInputRef = useRef(null);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [status, setStatus] = useState("idle"); // idle | uploading | error
  const [errorMessage, setErrorMessage] = useState("");

  const atLimit = attachmentCount >= MAX_ATTACHMENTS;

  // Closes the menu on any click outside it, same as a standard dropdown.
  useEffect(() => {
    if (!isMenuOpen) return;

    const handleOutsideClick = (event) => {
      if (!containerRef.current?.contains(event.target)) {
        setIsMenuOpen(false);
      }
    };

    document.addEventListener("mousedown", handleOutsideClick);
    return () => document.removeEventListener("mousedown", handleOutsideClick);
  }, [isMenuOpen]);

  // The error badge is transient - clear it automatically so it doesn't
  // linger indefinitely. Re-running this effect on every status change means
  // starting a new upload cancels any pending clear.
  useEffect(() => {
    if (status !== "error") return;

    const timeoutId = setTimeout(() => {
      setStatus("idle");
      setErrorMessage("");
    }, ERROR_MESSAGE_DURATION_MS);

    return () => clearTimeout(timeoutId);
  }, [status]);

  const handleFilesOptionClick = () => {
    setIsMenuOpen(false);
    fileInputRef.current?.click();
  };

  // Photos isn't implemented yet - clicking it only closes the menu.
  const handlePhotosOptionClick = () => {
    setIsMenuOpen(false);
  };

  const handleFileChange = async (event) => {
    const file = event.target.files?.[0];
    // Reset so selecting the same file again still fires onChange.
    event.target.value = "";
    if (!file || atLimit) return;

    // Client-side guard - backend still enforces the real limit with HTTP 413.
    if (file.size > MAX_DOCUMENT_SIZE_BYTES) {
      setStatus("error");
      setErrorMessage(`Maximum document size is ${MAX_DOCUMENT_SIZE_MB} MB.`);
      return;
    }

    setStatus("uploading");
    setErrorMessage("");

    try {
      const result = await uploadDocument(file, ragOptions);
      onUploaded({
        documentId: result.document_id,
        filename: result.filename,
        indexedChunkCount: result.indexed_chunk_count,
        ragOptions: ragOptions ? { ...ragOptions } : null,
      });
      setStatus("idle");
    } catch (error) {
      setStatus("error");
      setErrorMessage(error.message || "Upload failed. Please try again.");
    }
  };

  return (
    <div className="document-upload" ref={containerRef}>
      <input
        ref={fileInputRef}
        type="file"
        accept={ACCEPTED_FILE_TYPES}
        onChange={handleFileChange}
        className="document-upload-input"
      />

      <button
        type="button"
        className="document-upload-trigger"
        onClick={() => setIsMenuOpen((open) => !open)}
        disabled={status === "uploading" || atLimit}
        aria-label="Upload document"
        title={atLimit ? `You can attach up to ${MAX_ATTACHMENTS} files` : "Upload document"}
      >
        {status === "uploading" ? (
          <img src={spinnerIcon} alt="" className="document-upload-spinner" />
        ) : (
          "+"
        )}
      </button>

      {isMenuOpen && (
        <div className="document-upload-menu">
          <button type="button" onClick={handleFilesOptionClick}>
            Files
          </button>
          <button type="button" onClick={handlePhotosOptionClick}>
            Photos
          </button>
        </div>
      )}

      {status === "error" && (
        <span className="document-upload-status document-upload-status-error">
          {errorMessage}
        </span>
      )}
    </div>
  );
}

export default DocumentUpload;
