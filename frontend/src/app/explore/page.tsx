"use client";

import { useEffect, useState } from "react";
import { GitBranch, Database } from "lucide-react";
import { getSystemHealth } from "@/lib/api";
import type { SystemHealth } from "@/lib/types";

export default function ExplorePage() {
  const [health, setHealth] = useState<SystemHealth | null>(null);

  useEffect(() => {
    getSystemHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-1">Explore Data</h1>
      <p className="text-sm text-[var(--text-secondary)] mb-6">
        Inspect knowledge graphs and system statistics
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Knowledge Graphs */}
        <div className="border border-[var(--border)] rounded-xl bg-[var(--bg-secondary)] p-5">
          <div className="flex items-center gap-2 mb-4">
            <GitBranch size={20} className="text-amber-400" />
            <h2 className="font-semibold">Knowledge Graphs</h2>
          </div>
          {health?.knowledge_graphs?.length ? (
            <ul className="space-y-2">
              {health.knowledge_graphs.map((name) => (
                <li key={name} className="flex items-center gap-2 px-3 py-2 bg-[var(--bg-tertiary)] rounded-lg text-sm">
                  <GitBranch size={14} className="text-amber-400" />
                  {name}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-[var(--text-secondary)]">No knowledge graphs built yet</p>
          )}
        </div>

        {/* System Stats */}
        <div className="border border-[var(--border)] rounded-xl bg-[var(--bg-secondary)] p-5">
          <div className="flex items-center gap-2 mb-4">
            <Database size={20} className="text-blue-400" />
            <h2 className="font-semibold">System Stats</h2>
          </div>
          <div className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="text-[var(--text-secondary)]">Total Documents</span>
              <span>{health?.total_documents ?? 0}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--text-secondary)]">Ready</span>
              <span className="text-green-400">{health?.documents_ready ?? 0}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--text-secondary)]">Ingesting</span>
              <span className="text-yellow-400">{health?.documents_ingesting ?? 0}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--text-secondary)]">Available RAM</span>
              <span>{health?.available_ram_gb ?? "??"} GB</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--text-secondary)]">Ollama</span>
              <span className={health?.ollama_connected ? "text-green-400" : "text-red-400"}>
                {health?.ollama_connected ? "Connected" : "Offline"}
              </span>
            </div>
            {health?.ollama_models?.length ? (
              <div>
                <span className="text-[var(--text-secondary)]">Models:</span>
                <div className="mt-1 flex flex-wrap gap-1">
                  {health.ollama_models.map((m) => (
                    <span key={m} className="text-xs px-2 py-0.5 bg-[var(--bg-tertiary)] rounded">{m}</span>
                  ))}
                </div>
              </div>
            ) : null}
            {health?.cag_status && (
              <div className="pt-2 border-t border-[var(--border)]">
                <span className="text-[var(--text-secondary)]">CAG:</span>
                <span className="ml-2">
                  {(health.cag_status as Record<string, unknown>).active ? (
                    <span className="text-green-400">Active ({String((health.cag_status as Record<string, unknown>).document)})</span>
                  ) : (
                    <span className="text-[var(--text-secondary)]">Inactive</span>
                  )}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
