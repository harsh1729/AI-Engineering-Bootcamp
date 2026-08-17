import { useEffect, useRef, useState } from "react";
import { uploadDocument } from "../api/documentApi";
import { useAuth } from "../context/AuthContext";
import {
  MAX_ATTACHMENTS,
  MAX_DOCUMENT_SIZE_BYTES,
  MAX_DOCUMENT_SIZE_MB,
  MAX_IMAGE_SIZE_BYTES,
  MAX_IMAGE_SIZE_MB,
} from "../constants/attachments";
import spinnerIcon from "../assets/spinner.svg";

// Keep in sync with backend app.config.ALLOWED_DOCUMENT_EXTENSIONS
// (modern formats only - legacy .doc/.xls/.ppt are rejected server-side if enabled).
const ACCEPTED_FILE_TYPES =
  ".pdf,.txt,.docx,.xlsx,.pptx,.rtf,application/pdf,text/plain," +
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document," +
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet," +
  "application/vnd.openxmlformats-officedocument.presentationml.presentation," +
  "application/rtf,text/rtf";
// Keep in sync with backend app.config.ALLOWED_IMAGE_EXTENSIONS
const ACCEPTED_PHOTO_TYPES = "image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp";
const ERROR_MESSAGE_DURATION_MS = 6000;

function PaperclipIcon() {
  return (
    <svg
      className="document-upload-trigger-icon"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M8.5 12.5L14.2 6.8C15.6 5.4 17.8 5.4 19.2 6.8C20.6 8.2 20.6 10.4 19.2 11.8L11.8 19.2C9.7 21.3 6.3 21.3 4.2 19.2C2.1 17.1 2.1 13.7 4.2 11.6L12.5 3.3"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function isImageUpload(file) {
  if (!file) return false;
  if (file.type.startsWith("image/")) return true;
  const extension = file.name.includes(".")
    ? file.name.slice(file.name.lastIndexOf(".")).toLowerCase()
    : "";
  return [".jpg", ".jpeg", ".png", ".webp"].includes(extension);
}

function DocumentUpload({ attachmentCount, onUploaded, ragOptions }) {
  const { user } = useAuth();
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

  const handleUpload = async (file) => {
    if (!file || atLimit) return;

    const imageUpload = isImageUpload(file);
    const maxBytes = imageUpload ? MAX_IMAGE_SIZE_BYTES : MAX_DOCUMENT_SIZE_BYTES;
    const maxMb = imageUpload ? MAX_IMAGE_SIZE_MB : MAX_DOCUMENT_SIZE_MB;
    const sizeLabel = imageUpload ? "image" : "document";

    if (file.size > maxBytes) {
      setStatus("error");
      setErrorMessage(`Maximum ${sizeLabel} size is ${maxMb} MB.`);
      return;
    }

    setStatus("uploading");
    setErrorMessage("");

    try {
      const result = await uploadDocument(file, ragOptions, {
        useAuth: Boolean(user),
      });
      onUploaded({
        documentId: result.document_id,
        filename: result.filename,
        indexedChunkCount: result.indexed_chunk_count,
        ragOptions: ragOptions ? { ...ragOptions } : null,
        contentType: imageUpload ? "image" : "text",
      });
      setStatus("idle");
    } catch (error) {
      setStatus("error");
      setErrorMessage(error.message || "Upload failed. Please try again.");
    }
  };

  const handleFileChange = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    await handleUpload(file);
  };

  const handlePhotoChange = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    await handleUpload(file);
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
        className="message-input-action document-upload-trigger"
        onClick={handleTriggerClick}
        disabled={status === "uploading" || atLimit}
        aria-label="Attach document"
        aria-expanded={isMenuOpen}
        title={atLimit ? `You can attach up to ${MAX_ATTACHMENTS} files` : "Attach document"}
      >
        {status === "uploading" ? (
          <img src={spinnerIcon} alt="" className="document-upload-spinner" />
        ) : (
          <PaperclipIcon />
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
