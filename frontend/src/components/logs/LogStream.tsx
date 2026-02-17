"use client";

import { useRef, useEffect, useState } from "react";
import { Pause, Play, Trash2, Circle } from "lucide-react";
import type { LogEntry } from "@/lib/types";

interface LogStreamProps {
  logs: LogEntry[];
  connected: boolean;
  onClear: () => void;
}

const LEVEL_COLORS: Record<string, string> = {
  INFO: "text-blue-400",
  WARNING: "text-yellow-400",
  ERROR: "text-red-400",
  DEBUG: "text-gray-500",
};

const MODULE_COLORS: Record<string, string> = {
  extraction: "text-green-400",
  edi: "text-cyan-400",
  graph: "text-amber-400",
  router: "text-blue-400",
  grader: "text-orange-400",
  generator: "text-purple-400",
  cag: "text-pink-400",
  ingestion: "text-indigo-400",
  system: "text-gray-400",
};

export default function LogStream({ logs, connected, onClear }: LogStreamProps) {
  const [autoscroll, setAutoscroll] = useState(true);
  const [filter, setFilter] = useState("");
  const [levelFilter, setLevelFilter] = useState<string>("");
  const [moduleFilter, setModuleFilter] = useState<string>("");
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (autoscroll && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs, autoscroll]);

  const filteredLogs = logs.filter((log) => {
    if (filter && !log.message.toLowerCase().includes(filter.toLowerCase())) return false;
    if (levelFilter && log.level !== levelFilter) return false;
    if (moduleFilter && log.module !== moduleFilter) return false;
    return true;
  });

  const uniqueModules = [...new Set(logs.map((l) => l.module))];

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-[var(--border)] bg-[var(--bg-secondary)]">
        <div className="flex items-center gap-1.5">
          <Circle size={8} className={connected ? "fill-green-400 text-green-400" : "fill-red-400 text-red-400"} />
          <span className="text-xs text-[var(--text-secondary)]">{connected ? "Connected" : "Disconnected"}</span>
        </div>
        <input
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Search logs..."
          className="px-2 py-1 text-xs bg-[var(--bg-tertiary)] border border-[var(--border)] rounded text-[var(--text-primary)] w-48"
        />
        <select
          value={levelFilter}
          onChange={(e) => setLevelFilter(e.target.value)}
          className="text-xs bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-2 py-1 text-[var(--text-primary)]"
        >
          <option value="">All Levels</option>
          <option value="INFO">INFO</option>
          <option value="WARNING">WARNING</option>
          <option value="ERROR">ERROR</option>
        </select>
        <select
          value={moduleFilter}
          onChange={(e) => setModuleFilter(e.target.value)}
          className="text-xs bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-2 py-1 text-[var(--text-primary)]"
        >
          <option value="">All Modules</option>
          {uniqueModules.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
        <div className="flex-1" />
        <button
          onClick={() => setAutoscroll(!autoscroll)}
          className={`p-1 rounded ${autoscroll ? "text-[var(--accent)]" : "text-[var(--text-secondary)]"}`}
          title={autoscroll ? "Pause auto-scroll" : "Resume auto-scroll"}
        >
          {autoscroll ? <Pause size={14} /> : <Play size={14} />}
        </button>
        <button onClick={onClear} className="p-1 rounded text-[var(--text-secondary)] hover:text-[var(--error)]" title="Clear logs">
          <Trash2 size={14} />
        </button>
      </div>

      {/* Log output */}
      <div ref={containerRef} className="flex-1 overflow-y-auto bg-[var(--bg-primary)] p-3 font-mono text-xs leading-5">
        {filteredLogs.length === 0 && (
          <div className="text-[var(--text-secondary)] text-center py-8">
            {logs.length === 0 ? "Waiting for log events..." : "No matching logs"}
          </div>
        )}
        {filteredLogs.map((log, i) => (
          <div key={i} className="flex gap-2 hover:bg-[var(--bg-tertiary)] px-1 rounded">
            <span className="text-[var(--text-secondary)] shrink-0">
              {new Date(log.timestamp).toLocaleTimeString()}
            </span>
            <span className={`shrink-0 w-12 text-right ${LEVEL_COLORS[log.level] || "text-[var(--text-secondary)]"}`}>
              {log.level}
            </span>
            <span className={`shrink-0 w-20 ${MODULE_COLORS[log.module] || "text-[var(--text-secondary)]"}`}>
              [{log.module}]
            </span>
            {log.document_id && (
              <span className="text-[var(--text-secondary)] shrink-0">#{log.document_id}</span>
            )}
            <span className="text-[var(--text-primary)]">{log.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
