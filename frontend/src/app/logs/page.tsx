"use client";

import { useLogStream } from "@/lib/websocket";
import LogStream from "@/components/logs/LogStream";

export default function LogsPage() {
  const { logs, connected, clearLogs } = useLogStream();

  return (
    <div className="h-full flex flex-col">
      <div className="px-6 py-4 border-b border-[var(--border)]">
        <h1 className="text-2xl font-bold">System Logs</h1>
        <p className="text-sm text-[var(--text-secondary)] mt-1">
          Real-time log stream from the backend ingestion and retrieval pipelines
        </p>
      </div>
      <div className="flex-1">
        <LogStream logs={logs} connected={connected} onClear={clearLogs} />
      </div>
    </div>
  );
}
