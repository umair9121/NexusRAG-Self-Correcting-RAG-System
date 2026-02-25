"use client";

import { useState } from "react";
import { X, Eraser, Loader2 } from "lucide-react";
import {
  deleteDocumentMetadata,
  METADATA_KEYS,
  type MetadataKey,
} from "@/lib/api";

const METADATA_LABELS: Record<string, string> = {
  pageindex: "Page Index",
  docstore: "Doc Store",
  chroma: "Vector Embeddings",
  graph: "Knowledge Graph",
  communities: "Communities",
  figures: "Figures",
};

interface ClearMetadataModalProps {
  docId: number;
  docName: string;
  open: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

export default function ClearMetadataModal({
  docId,
  docName,
  open,
  onClose,
  onSuccess,
}: ClearMetadataModalProps) {
  const [clearing, setClearing] = useState(false);
  const [selected, setSelected] = useState<Set<MetadataKey>>(new Set());

  if (!open) return null;

  const toggleKey = (key: MetadataKey) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const handleClearAll = async () => {
    setClearing(true);
    try {
      await deleteDocumentMetadata(docId);
      onSuccess();
      onClose();
    } catch (e) {
      console.error(e);
      alert(e instanceof Error ? e.message : "Failed to clear metadata");
    } finally {
      setClearing(false);
    }
  };

  const handleClearSelected = async () => {
    if (selected.size === 0) return;
    setClearing(true);
    try {
      await deleteDocumentMetadata(docId, Array.from(selected));
      onSuccess();
      onClose();
    } catch (e) {
      console.error(e);
      alert(e instanceof Error ? e.message : "Failed to clear metadata");
    } finally {
      setClearing(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative w-full max-w-md bg-[var(--bg-secondary)] rounded-xl border border-[var(--border)] shadow-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-base font-semibold text-[var(--text-primary)]">
            Clear metadata
          </h3>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-[var(--bg-tertiary)] text-[var(--text-secondary)]"
            aria-label="Close"
          >
            <X size={18} />
          </button>
        </div>
        <p className="text-sm text-[var(--text-secondary)] mb-1 truncate" title={docName}>
          {docName}
        </p>
        <p className="text-xs text-[var(--text-secondary)] mb-4">
          The document file will not be deleted. Choose what to remove.
        </p>

        <div className="space-y-3">
          <button
            onClick={handleClearAll}
            disabled={clearing}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-amber-600/90 text-white hover:bg-amber-600 disabled:opacity-50 text-sm font-medium"
          >
            {clearing ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <Eraser size={16} />
            )}
            Clear all metadata
          </button>

          <div className="border-t border-[var(--border)] pt-3">
            <p className="text-xs font-medium text-[var(--text-secondary)] mb-2">
              Or clear selected:
            </p>
            <div className="grid grid-cols-2 gap-2 mb-3">
              {METADATA_KEYS.map((key) => (
                <label
                  key={key}
                  className="flex items-center gap-2 cursor-pointer text-sm text-[var(--text-primary)]"
                >
                  <input
                    type="checkbox"
                    checked={selected.has(key as MetadataKey)}
                    onChange={() => toggleKey(key as MetadataKey)}
                    className="accent-[var(--accent)] rounded"
                  />
                  {METADATA_LABELS[key] ?? key}
                </label>
              ))}
            </div>
            <button
              onClick={handleClearSelected}
              disabled={clearing || selected.size === 0}
              className="w-full px-4 py-2 rounded-lg bg-[var(--bg-tertiary)] hover:bg-[var(--border)] text-[var(--text-primary)] text-sm disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Clear selected ({selected.size})
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
