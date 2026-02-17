"use client";

import { ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";

interface TraceViewerProps {
  trace: string[];
  metadata?: Record<string, unknown>;
}

const STEP_COLORS: Record<string, string> = {
  router: "border-l-blue-500",
  grader: "border-l-orange-500",
  error: "border-l-red-500",
  generator: "border-l-purple-500",
  pageindex: "border-l-green-500",
  vector: "border-l-cyan-500",
  graph: "border-l-amber-500",
  cag: "border-l-pink-500",
  rewriter: "border-l-yellow-500",
};

function getStepColor(step: string): string {
  const lower = step.toLowerCase();
  for (const [key, color] of Object.entries(STEP_COLORS)) {
    if (lower.includes(key)) return color;
  }
  return "border-l-gray-500";
}

export default function TraceViewer({ trace, metadata }: TraceViewerProps) {
  const [expanded, setExpanded] = useState(false);

  if (!trace?.length) return null;

  return (
    <div className="mt-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1 text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
      >
        {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        Reasoning Trace ({trace.length} steps)
      </button>
      {expanded && (
        <div className="mt-2 space-y-1">
          {trace.map((step, i) => (
            <div
              key={i}
              className={`pl-3 py-1 border-l-2 text-xs font-mono text-[var(--text-secondary)] ${getStepColor(step)}`}
            >
              {step}
            </div>
          ))}
          {metadata && Object.keys(metadata).length > 0 && (
            <details className="mt-2">
              <summary className="text-xs text-[var(--text-secondary)] cursor-pointer">Metadata</summary>
              <pre className="mt-1 p-2 text-xs bg-[var(--bg-tertiary)] rounded overflow-auto max-h-32">
                {JSON.stringify(metadata, null, 2)}
              </pre>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
