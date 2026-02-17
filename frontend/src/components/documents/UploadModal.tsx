"use client";

import { useState, useRef } from "react";
import { Upload, X, FileText, FolderOpen, FolderInput } from "lucide-react";
import { uploadDocuments, importFolderFromPath, ALLOWED_DOCUMENT_EXTENSIONS } from "@/lib/api";

interface UploadModalProps {
  open: boolean;
  onClose: () => void;
  onUploaded: () => void;
}

type Tab = "upload" | "folder-path";

export default function UploadModal({ open, onClose, onUploaded }: UploadModalProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [pathTags, setPathTags] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [tab, setTab] = useState<Tab>("upload");
  const inputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const [serverPath, setServerPath] = useState("");
  const [pathImportError, setPathImportError] = useState("");

  if (!open) return null;

  const isAllowed = (name: string) =>
    ALLOWED_DOCUMENT_EXTENSIONS.some((ext) => name.toLowerCase().endsWith(ext));

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const dropped = Array.from(e.dataTransfer.files).filter((f) => isAllowed(f.name));
    setFiles((prev) => [...prev, ...dropped]);
    setPathTags((prev) => prev);
  };

  const handleSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const list = Array.from(e.target.files).filter((f) => isAllowed(f.name));
      setFiles((prev) => [...prev, ...list]);
    }
    e.target.value = "";
  };

  const handleSelectFolder = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    const list = Array.from(e.target.files).filter((f) => isAllowed(f.name));
    const paths = list.map((f) => (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name);
    setFiles((prev) => [...prev, ...list]);
    setPathTags((prev) => [...prev, ...paths]);
    e.target.value = "";
  };

  const handleUpload = async () => {
    if (!files.length) return;
    setUploading(true);
    try {
      await uploadDocuments(files, pathTags.length === files.length ? pathTags : undefined);
      setFiles([]);
      setPathTags([]);
      onUploaded();
      onClose();
    } catch (err) {
      alert("Upload failed: " + (err as Error).message);
    } finally {
      setUploading(false);
    }
  };

  const handleImportFromPath = async () => {
    const path = serverPath.trim();
    if (!path) return;
    setPathImportError("");
    setUploading(true);
    try {
      await importFolderFromPath(path);
      setServerPath("");
      onUploaded();
      onClose();
    } catch (err) {
      setPathImportError((err as Error).message);
    } finally {
      setUploading(false);
    }
  };

  const removeFile = (i: number) => {
    setFiles((prev) => prev.filter((_, j) => j !== i));
    if (pathTags.length === files.length) {
      setPathTags((prev) => prev.filter((_, j) => j !== i));
    }
  };

  const isFolderUpload = pathTags.length === files.length && files.length > 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
      <div
        className="w-full max-w-lg bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Add Documents</h2>
          <button onClick={onClose} className="text-[var(--text-secondary)] hover:text-white">
            <X size={20} />
          </button>
        </div>

        <div className="flex gap-2 mb-4 border-b border-[var(--border)]">
          <button
            type="button"
            onClick={() => setTab("upload")}
            className={`px-3 py-2 text-sm rounded-t ${tab === "upload" ? "bg-[var(--bg-tertiary)] text-white" : "text-[var(--text-secondary)] hover:text-white"}`}
          >
            Upload files or folder
          </button>
          <button
            type="button"
            onClick={() => setTab("folder-path")}
            className={`px-3 py-2 text-sm rounded-t ${tab === "folder-path" ? "bg-[var(--bg-tertiary)] text-white" : "text-[var(--text-secondary)] hover:text-white"}`}
          >
            Connect server folder
          </button>
        </div>

        {tab === "upload" && (
          <>
            <div
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              className="border-2 border-dashed border-[var(--border)] rounded-lg p-6 text-center cursor-pointer hover:border-[var(--accent)] transition-colors"
            >
              <Upload size={28} className="mx-auto mb-2 text-[var(--text-secondary)]" />
              <p className="text-sm text-[var(--text-secondary)] mb-2">Drop documents here or pick files (PDF, Word, Excel, JSON, HTML, images)</p>
              <div className="flex items-center justify-center gap-3 flex-wrap">
                <input ref={inputRef} type="file" accept={ALLOWED_DOCUMENT_EXTENSIONS.join(",")} multiple hidden onChange={handleSelect} />
                <button
                  type="button"
                  onClick={() => inputRef.current?.click()}
                  className="px-3 py-1.5 text-sm rounded-lg border border-[var(--border)] hover:bg-[var(--bg-tertiary)]"
                >
                  Select files
                </button>
                <input
                  ref={folderInputRef}
                  type="file"
                  accept={ALLOWED_DOCUMENT_EXTENSIONS.join(",")}
                  multiple
                  hidden
                  onChange={handleSelectFolder}
                  {...{ directory: "", webkitdirectory: "" }}
                />
                <button
                  type="button"
                  onClick={() => folderInputRef.current?.click()}
                  className="px-3 py-1.5 text-sm rounded-lg border border-[var(--border)] hover:bg-[var(--bg-tertiary)] flex items-center gap-1.5"
                >
                  <FolderOpen size={16} />
                  Select folder
                </button>
              </div>
              <p className="text-xs text-[var(--text-secondary)] mt-2">Folder: path is stored as metadata/tags</p>
            </div>

            {files.length > 0 && (
              <div className="mt-4 max-h-40 overflow-y-auto space-y-1">
                {files.map((f, i) => (
                  <div key={i} className="flex items-center gap-2 text-sm px-2 py-1 bg-[var(--bg-tertiary)] rounded">
                    <FileText size={14} />
                    <span className="flex-1 truncate" title={pathTags[i] || f.name}>
                      {pathTags[i] || f.name}
                    </span>
                    <span className="text-xs text-[var(--text-secondary)] shrink-0">{(f.size / 1024).toFixed(0)} KB</span>
                    <button
                      type="button"
                      onClick={() => removeFile(i)}
                      className="text-[var(--text-secondary)] hover:text-[var(--error)]"
                    >
                      <X size={14} />
                    </button>
                  </div>
                ))}
              </div>
            )}

            <div className="mt-4 flex justify-end gap-2">
              <button type="button" onClick={onClose} className="px-4 py-2 text-sm rounded-lg border border-[var(--border)] hover:bg-[var(--bg-tertiary)]">
                Cancel
              </button>
              <button
                type="button"
                onClick={handleUpload}
                disabled={!files.length || uploading}
                className="px-4 py-2 text-sm rounded-lg bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] disabled:opacity-50"
              >
                {uploading ? "Uploading..." : `Upload ${files.length} file(s)${isFolderUpload ? " (with paths)" : ""}`}
              </button>
            </div>
          </>
        )}

        {tab === "folder-path" && (
          <>
            <p className="text-sm text-[var(--text-secondary)] mb-2">
              Import all supported documents from a folder on the server. Path will be used as metadata/tags.
            </p>
            <div className="flex gap-2">
              <input
                type="text"
                value={serverPath}
                onChange={(e) => setServerPath(e.target.value)}
                placeholder="e.g. C:\Users\You\Documents\MyFolder or /home/user/docs"
                className="flex-1 px-3 py-2 rounded-lg border border-[var(--border)] bg-[var(--bg-tertiary)] text-sm"
              />
              <button
                type="button"
                onClick={handleImportFromPath}
                disabled={!serverPath.trim() || uploading}
                className="px-4 py-2 text-sm rounded-lg bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] disabled:opacity-50 flex items-center gap-1.5"
              >
                <FolderInput size={16} />
                {uploading ? "Importing..." : "Import"}
              </button>
            </div>
            {pathImportError && (
              <p className="mt-2 text-sm text-[var(--error)]">{pathImportError}</p>
            )}
            <div className="mt-4 flex justify-end">
              <button type="button" onClick={onClose} className="px-4 py-2 text-sm rounded-lg border border-[var(--border)] hover:bg-[var(--bg-tertiary)]">
                Close
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
