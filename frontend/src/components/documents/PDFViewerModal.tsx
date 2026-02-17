"use client";

import { useEffect, useRef, useState } from "react";
import { X, Maximize2, Minimize2, Download, ExternalLink, Loader2, Columns } from "lucide-react";

interface PDFViewerModalProps {
  open: boolean;
  onClose: () => void;
  pdfUrl: string;
  documentName: string;
  onOpenSideBySide?: () => void;
}

export default function PDFViewerModal({
  open,
  onClose,
  pdfUrl,
  documentName,
  onOpenSideBySide,
}: PDFViewerModalProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (open) {
      setLoading(true);
    }
  }, [open, pdfUrl]);

  useEffect(() => {
    if (!open) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [open, onClose]);

  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!isFullscreen) {
      containerRef.current.requestFullscreen?.();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen?.();
      setIsFullscreen(false);
    }
  };

  useEffect(() => {
    const handler = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", handler);
    return () => document.removeEventListener("fullscreenchange", handler);
  }, []);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-xl"
        onClick={onClose}
      />

      {/* Modal */}
      <div
        ref={containerRef}
        className={`relative flex flex-col overflow-hidden transition-all duration-300 ${
          isFullscreen
            ? "w-screen h-screen rounded-none"
            : "w-[92vw] max-w-6xl h-[92vh] rounded-2xl"
        }`}
        style={{
          background:
            "linear-gradient(135deg, rgba(17,17,24,0.92) 0%, rgba(10,10,15,0.96) 100%)",
          border: "1px solid rgba(99,102,241,0.15)",
          boxShadow:
            "0 0 0 1px rgba(255,255,255,0.03), 0 25px 80px -12px rgba(0,0,0,0.8), 0 0 60px -15px rgba(99,102,241,0.15), inset 0 1px 0 rgba(255,255,255,0.05)",
        }}
      >
        {/* Glassy header */}
        <div
          className="shrink-0 flex items-center justify-between px-5 py-3 select-none"
          style={{
            background:
              "linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%)",
            borderBottom: "1px solid rgba(255,255,255,0.06)",
            backdropFilter: "blur(20px)",
          }}
        >
          <div className="flex items-center gap-3 min-w-0 flex-1">
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
              style={{
                background:
                  "linear-gradient(135deg, rgba(99,102,241,0.2) 0%, rgba(99,102,241,0.08) 100%)",
                border: "1px solid rgba(99,102,241,0.2)",
              }}
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="text-indigo-400"
              >
                <path d="M14.5 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V7.5L14.5 2z" />
                <polyline points="14 2 14 8 20 8" />
              </svg>
            </div>
            <h2 className="text-sm font-semibold text-[var(--text-primary)] truncate">
              {documentName}
            </h2>
          </div>

          <div className="flex items-center gap-0.5 ml-4">
            {onOpenSideBySide && (
              <>
                <button
                  onClick={onOpenSideBySide}
                  className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium bg-indigo-500/10 text-indigo-400 hover:bg-indigo-500/20 transition-all active:scale-95"
                  title="View PDF and Metadata side by side"
                >
                  <Columns size={13} />
                  <span>Side by Side</span>
                </button>
                <div className="w-px h-5 bg-white/8 mx-1" />
              </>
            )}
            <a
              href={pdfUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="p-1.5 rounded-lg hover:bg-white/5 transition-all active:scale-95"
              title="Open in new tab"
            >
              <ExternalLink size={14} className="text-[var(--text-secondary)]" />
            </a>
            <a
              href={pdfUrl}
              download={documentName}
              className="p-1.5 rounded-lg hover:bg-white/5 transition-all active:scale-95"
              title="Download PDF"
            >
              <Download size={14} className="text-[var(--text-secondary)]" />
            </a>
            <button
              onClick={toggleFullscreen}
              className="p-1.5 rounded-lg hover:bg-white/5 transition-all active:scale-95"
              title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
            >
              {isFullscreen ? (
                <Minimize2 size={14} className="text-[var(--text-secondary)]" />
              ) : (
                <Maximize2 size={14} className="text-[var(--text-secondary)]" />
              )}
            </button>
            <div className="w-px h-5 bg-white/8 mx-1.5" />
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-red-500/10 transition-all active:scale-95 group"
              title="Close (Esc)"
            >
              <X size={15} className="text-[var(--text-secondary)] group-hover:text-red-400 transition-colors" />
            </button>
          </div>
        </div>

        {/* PDF iframe – browser provides scroll, Ctrl+F, zoom, rotate */}
        <div
          className="flex-1 relative min-h-0"
          style={{
            background:
              "radial-gradient(ellipse at center, rgba(30,30,45,0.5) 0%, rgba(10,10,15,1) 70%)",
          }}
        >
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center z-10">
              <div
                className="flex flex-col items-center gap-3"
                style={{
                  background:
                    "linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(99,102,241,0.05) 100%)",
                  border: "1px solid rgba(99,102,241,0.15)",
                  borderRadius: "1rem",
                  padding: "1.5rem",
                }}
              >
                <Loader2 size={22} className="animate-spin text-indigo-400" />
                <span className="text-sm text-[var(--text-secondary)]">
                  Loading PDF...
                </span>
              </div>
            </div>
          )}
          <iframe
            src={pdfUrl}
            title={documentName}
            className="w-full h-full rounded-b-2xl border-0"
            style={{ borderBottomLeftRadius: "1rem", borderBottomRightRadius: "1rem" }}
            onLoad={() => setLoading(false)}
          />
        </div>
      </div>
    </div>
  );
}
