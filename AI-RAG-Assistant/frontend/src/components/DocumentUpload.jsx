import { useEffect, useRef, useState } from "react";
import { uploadDocument } from "../api/documentApi";
import { MAX_ATTACHMENTS, MAX_DOCUMENT_SIZE_BYTES, MAX_DOCUMENT_SIZE_MB } from "../constants/attachments";
import spinnerIcon from "../assets/spinner.svg";

// Keep in sync with backend app.config.ALLOWED_DOCUMENT_EXTENSIONS
// (modern formats only - legacy .doc/.xls/.ppt are rejected server-side if enabled).
const ACCEPTED_FILE_TYPES =
  ".pdf,.txt,.docx,.xlsx,.pptx,.rtf,application/pdf,text/plain," +
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document," +
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet," +
  "application/vnd.openxmlformats-officedocument.presentationml.presentation," +
  "application/rtf,text/rtf";
const ACCEPTED_PHOTO_TYPES = "image/*";
const PHOTO_UPLOAD_UNSUPPORTED_MESSAGE =
  "Photo upload isn't supported yet. Use Files to attach PDF, Word, or text documents.";
const ERROR_MESSAGE_DURATION_MS = 6000;

function DocumentUpload({ attachmentCount, onUploaded, ragOptions }) {
  const containerRef = useRef(null);
  const fileInputRef = useRef(null);
  const photoInputRef = useRef(null);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [status, setStatus] = useState("idle"); // idle | uploading | error
  const [errorMessage, setErrorMessage] = useState("");

  const atLimit = attachmentCount >= MAX_ATTACHMENTS;

  useEffect(() => {
    if (!isMenuOpen) return;

    const handleOutsideClick = (event) => {
      if (!containerRef.current?.contains(event.target)) {
        setIsMenuOpen(false);
      }
    };

    document.addEventListener("mousedown", handleOutsideClick);
    document.addEventListener("touchstart", handleOutsideClick);
    return () => {
      document.removeEventListener("mousedown", handleOutsideClick);
      document.removeEventListener("touchstart", handleOutsideClick);
    };
  }, [isMenuOpen]);

  useEffect(() => {
    if (status !== "error") return;

    const timeoutId = setTimeout(() => {
      setStatus("idle");
      setErrorMessage("");
    }, ERROR_MESSAGE_DURATION_MS);

    return () => clearTimeout(timeoutId);
  }, [status]);

  const openFilePicker = () => {
    fileInputRef.current?.click();
  };

  const openPhotoPicker = () => {
    photoInputRef.current?.click();
  };

  const handleFilesOptionClick = () => {
    setIsMenuOpen(false);
    openFilePicker();
  };

  const handlePhotosOptionClick = () => {
    setIsMenuOpen(false);
    openPhotoPicker();
  };

  const handleTriggerClick = () => {
    if (status === "uploading" || atLimit) {
      return;
    }

    setIsMenuOpen((open) => !open);
  };

  const handleFileChange = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file || atLimit) return;

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

  const handlePhotoChange = (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    setStatus("error");
    setErrorMessage(PHOTO_UPLOAD_UNSUPPORTED_MESSAGE);
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

      <input
        ref={photoInputRef}
        type="file"
        accept={ACCEPTED_PHOTO_TYPES}
        onChange={handlePhotoChange}
        className="document-upload-input"
      />

      <button
        type="button"
        className="document-upload-trigger"
        onClick={handleTriggerClick}
        disabled={status === "uploading" || atLimit}
        aria-label="Upload document"
        aria-expanded={isMenuOpen}
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
