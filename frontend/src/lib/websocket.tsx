"use client";

import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  useCallback,
} from "react";
import type { LogEntry } from "./types";

// Use same host/port as API so logs work when backend runs on non-default port
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
const WS_URL =
  process.env.NEXT_PUBLIC_WS_URL ||
  `${API_BASE.replace(/^http/, "ws").replace(/\/?$/, "")}/ws/logs`;
const MAX_LOGS = 1000;

function logKey(entry: LogEntry): string {
  return `${entry.timestamp}\n${entry.module}\n${entry.message}\n${entry.document_id ?? ""}`;
}

/* ── Context ─────────────────────────────────────────────────────────────── */

interface LogStreamValue {
  logs: LogEntry[];
  connected: boolean;
  clearLogs: () => void;
}

const LogStreamContext = createContext<LogStreamValue | null>(null);

/**
 * Consume the persistent log stream from anywhere in the component tree.
 * Must be used inside <LogStreamProvider>.
 */
export function useLogStream(): LogStreamValue {
  const ctx = useContext(LogStreamContext);
  if (!ctx) {
    throw new Error("useLogStream must be used within <LogStreamProvider>");
  }
  return ctx;
}

/* ── Provider (mount once in the root layout) ────────────────────────────── */

export function LogStreamProvider({ children }: { children: React.ReactNode }) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();
  const seenKeysRef = useRef<Set<string>>(new Set());

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      };

      ws.onmessage = (event) => {
        try {
          const entry: LogEntry = JSON.parse(event.data);
          const key = logKey(entry);
          setLogs((prev) => {
            if (seenKeysRef.current.has(key)) return prev;
            seenKeysRef.current.add(key);
            const next = [...prev.slice(-(MAX_LOGS - 1)), entry];
            if (seenKeysRef.current.size > MAX_LOGS * 2) {
              const keep = new Set(next.map(logKey));
              seenKeysRef.current = keep;
            }
            return next;
          });
        } catch {
          // ignore malformed messages
        }
      };

      ws.onclose = () => {
        setConnected(false);
        wsRef.current = null;
        reconnectTimer.current = setTimeout(connect, 3000);
      };

      ws.onerror = () => {
        ws.close();
      };
    } catch {
      reconnectTimer.current = setTimeout(connect, 3000);
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect]);

  const clearLogs = useCallback(() => {
    seenKeysRef.current = new Set();
    setLogs([]);
  }, []);

  return (
    <LogStreamContext.Provider value={{ logs, connected, clearLogs }}>
      {children}
    </LogStreamContext.Provider>
  );
}
