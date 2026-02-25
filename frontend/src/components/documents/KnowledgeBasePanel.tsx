"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  Brain,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  Database,
  Link2,
  FileText,
  Tag,
  Search,
  GripHorizontal,
} from "lucide-react";
import {
  getKnowledgeBaseStats,
  refreshKnowledgeBase,
  getSystemHealth,
  listKnowledgeBaseEntities,
} from "@/lib/api";
import type { KnowledgeBaseStats, KnowledgeBaseEntity } from "@/lib/types";

const DEFAULT_KB_MODEL = "phi3:3.8b";

const STAT_META: Record<string, { label: string; description: string; icon: React.ReactNode }> = {
  entities: {
    label: "Entities",
    description: "Unique named things (people, places, components, etc.) extracted from your documents and classified by type.",
    icon: <Database size={14} />,
  },
  relations: {
    label: "Relations",
    description: "Subject–relationship–object facts linking two entities (e.g. “Engine powers Ship”).",
    icon: <Link2 size={14} />,
  },
  documents: {
    label: "Documents",
    description: "Source documents that contributed at least one entity or relation to the knowledge base.",
    icon: <FileText size={14} />,
  },
  types: {
    label: "Types",
    description: "Entity categories (e.g. manufacturer, vessel, component) used for filtering and disambiguation.",
    icon: <Tag size={14} />,
  },
};

const POPOVER_MIN_H = 160;
const POPOVER_MAX_H = 380;
const POPOVER_DEFAULT_H = 220;

export default function KnowledgeBasePanel() {
  const [stats, setStats] = useState<KnowledgeBaseStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [refreshMessage, setRefreshMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [openCard, setOpenCard] = useState<string | null>(null);
  const [popoverHeight, setPopoverHeight] = useState(POPOVER_DEFAULT_H);
  const [searchQuery, setSearchQuery] = useState("");
  const [topEntities, setTopEntities] = useState<KnowledgeBaseEntity[]>([]);
  const [topLoading, setTopLoading] = useState(false);
  const [searchResults, setSearchResults] = useState<KnowledgeBaseEntity[]>([]);
  const [searching, setSearching] = useState(false);
  const popoverRef = useRef<HTMLDivElement>(null);
  const dragStartY = useRef(0);
  const dragStartH = useRef(0);

  const fetchStats = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getKnowledgeBaseStats();
      setStats(data);
      setError(null);
    } catch {
      setError("Could not load knowledge base stats");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 30000);
    return () => clearInterval(interval);
  }, [fetchStats]);

  useEffect(() => {
    if (!expanded) return;
    getSystemHealth()
      .then((h) => h.ollama_models && setOllamaModels(h.ollama_models))
      .catch(() => {});
  }, [expanded]);

  // When dropdown opens, load top entities (by mentions) immediately
  useEffect(() => {
    if (!openCard) return;
    setTopLoading(true);
    setTopEntities([]);
    listKnowledgeBaseEntities({ limit: 25 })
      .then((r) => setTopEntities(r.entities))
      .catch(() => setTopEntities([]))
      .finally(() => setTopLoading(false));
  }, [openCard]);

  // When user types, narrow down with debounced search
  useEffect(() => {
    if (!openCard) return;
    const q = searchQuery.trim();
    if (!q) return;
    const t = setTimeout(() => {
      setSearching(true);
      listKnowledgeBaseEntities({ search: q, limit: 25 })
        .then((r) => setSearchResults(r.entities))
        .catch(() => setSearchResults([]))
        .finally(() => setSearching(false));
    }, 200);
    return () => clearTimeout(t);
  }, [openCard, searchQuery]);

  const handleDragStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    dragStartY.current = e.clientY;
    dragStartH.current = popoverHeight;
    const onMove = (e2: MouseEvent) => {
      const dy = e2.clientY - dragStartY.current;
      setPopoverHeight(Math.min(POPOVER_MAX_H, Math.max(POPOVER_MIN_H, dragStartH.current + dy)));
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, [popoverHeight]);

  const handleClickOutside = useCallback((e: MouseEvent) => {
    const target = e.target as HTMLElement;
    if (popoverRef.current?.contains(target) || target.closest("[data-kb-stat-card]")) return;
    setOpenCard(null);
  }, []);

  useEffect(() => {
    if (!openCard) return;
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [openCard, handleClickOutside]);

  const handleRefresh = async () => {
    setRefreshing(true);
    setRefreshMessage(null);
    try {
      const modelParam = selectedModel && selectedModel.trim() ? selectedModel.trim() : undefined;
      const result = await refreshKnowledgeBase(modelParam);
      setRefreshMessage(result.message);
      await fetchStats();
    } catch (e) {
      setRefreshMessage(`Refresh failed: ${e instanceof Error ? e.message : "Unknown error"}`);
    } finally {
      setRefreshing(false);
    }
  };

  const defaultModel = stats?.default_classification_model || DEFAULT_KB_MODEL;

  const isEmpty = !stats || stats.entity_count === 0;
  const typeEntries = stats ? Object.entries(stats.types).sort((a, b) => b[1] - a[1]) : [];

  return (
    <div className="mb-6 border border-[var(--border)] rounded-xl bg-[var(--bg-secondary)] overflow-hidden">
      {/* Header row — always visible */}
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center justify-between px-5 py-3.5 hover:bg-[var(--bg-primary)] transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-[var(--accent)]/10">
            <Brain size={18} className="text-[var(--accent)]" />
          </div>
          <div className="text-left">
            <h3 className="text-sm font-semibold text-[var(--text-primary)]">Knowledge Base</h3>
            <p className="text-xs text-[var(--text-secondary)]">
              {loading
                ? "Loading..."
                : isEmpty
                ? "No entities yet — ingest documents with Graph pipeline to populate"
                : `${stats.entity_count} entities · ${stats.relation_count} relations · ${stats.document_count} documents`}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {stats?.last_updated && (
            <span className="text-xs text-[var(--text-secondary)]">
              Updated {new Date(stats.last_updated).toLocaleString()}
            </span>
          )}
          {expanded ? <ChevronUp size={16} className="text-[var(--text-secondary)]" /> : <ChevronDown size={16} className="text-[var(--text-secondary)]" />}
        </div>
      </button>

      {/* Expanded detail section */}
      {expanded && (
        <div className="px-5 pb-4 border-t border-[var(--border)]">
          {error && (
            <div className="mt-3 text-sm text-red-500">{error}</div>
          )}

          {/* Stats grid — clickable cards with mini dropdown */}
          {!isEmpty && stats && (
            <div className="mt-4 relative overflow-visible">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {(["entities", "relations", "documents", "types"] as const).map((id) => {
                  const meta = STAT_META[id];
                  const value =
                    id === "entities"
                      ? stats.entity_count
                      : id === "relations"
                        ? stats.relation_count
                        : id === "documents"
                          ? stats.document_count
                          : Object.keys(stats.types).length;
                  return (
                    <StatCard
                      key={id}
                      id={id}
                      icon={meta.icon}
                      label={meta.label}
                      value={value}
                      isOpen={openCard === id}
                      onClick={() => {
                        setOpenCard((c) => (c === id ? null : id));
                        setSearchQuery("");
                        setSearchResults([]);
                      }}
                    />
                  );
                })}
              </div>
              {openCard && (
                <div
                  ref={popoverRef}
                  className="absolute left-0 right-0 top-full mt-2 z-10 rounded-lg border border-[var(--border)] bg-[var(--bg-secondary)] shadow-lg overflow-hidden flex flex-col"
                  style={{ height: popoverHeight }}
                >
                  <div className="px-3 py-2 border-b border-[var(--border)] shrink-0">
                    <h4 className="text-xs font-semibold text-[var(--text-primary)] uppercase tracking-wide mb-1">
                      {STAT_META[openCard]?.label}
                    </h4>
                    <p className="text-xs text-[var(--text-secondary)] leading-snug">
                      {STAT_META[openCard]?.description}
                    </p>
                  </div>
                  <div className="px-3 py-2 border-b border-[var(--border)] shrink-0">
                    <div className="relative">
                      <Search size={12} className="absolute left-2 top-1/2 -translate-y-1/2 text-[var(--text-secondary)]" />
                      <input
                        type="search"
                        placeholder="Search to narrow down..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full pl-7 pr-2 py-1.5 text-xs bg-[var(--bg-primary)] border border-[var(--border)] rounded-md text-[var(--text-primary)] placeholder:text-[var(--text-secondary)] focus:outline-none focus:border-[var(--accent)]"
                      />
                    </div>
                  </div>
                  <div className="flex-1 min-h-0 overflow-y-auto px-3 py-2">
                    {!searchQuery.trim() && topLoading && (
                      <p className="text-xs text-[var(--text-secondary)]">Loading top entities...</p>
                    )}
                    {!searchQuery.trim() && !topLoading && topEntities.length === 0 && (
                      <p className="text-xs text-[var(--text-secondary)]">No entities in the knowledge base.</p>
                    )}
                    {!searchQuery.trim() && !topLoading && topEntities.length > 0 && (
                      <ul className="space-y-1">
                        <p className="text-[10px] uppercase tracking-wide text-[var(--text-secondary)] mb-1.5">Top by mentions</p>
                        {topEntities.map((ent) => (
                          <li
                            key={ent.name}
                            className="flex items-center justify-between gap-2 py-1 px-2 rounded text-xs bg-[var(--bg-primary)]"
                          >
                            <span className="font-medium text-[var(--text-primary)] truncate">{ent.name}</span>
                            <span className="text-[var(--text-secondary)] shrink-0">
                              {ent.types[0]?.replace(/_/g, " ") ?? "—"} · {ent.mentions}×
                            </span>
                          </li>
                        ))}
                      </ul>
                    )}
                    {searchQuery.trim() && searching && (
                      <p className="text-xs text-[var(--text-secondary)]">Searching...</p>
                    )}
                    {searchQuery.trim() && !searching && searchResults.length === 0 && (
                      <p className="text-xs text-[var(--text-secondary)]">No entities match &quot;{searchQuery.trim()}&quot;.</p>
                    )}
                    {searchQuery.trim() && !searching && searchResults.length > 0 && (
                      <ul className="space-y-1">
                        <p className="text-[10px] uppercase tracking-wide text-[var(--text-secondary)] mb-1.5">Search results</p>
                        {searchResults.map((ent) => (
                          <li
                            key={ent.name}
                            className="flex items-center justify-between gap-2 py-1 px-2 rounded text-xs bg-[var(--bg-primary)]"
                          >
                            <span className="font-medium text-[var(--text-primary)] truncate">{ent.name}</span>
                            <span className="text-[var(--text-secondary)] shrink-0">
                              {ent.types[0]?.replace(/_/g, " ") ?? "—"} · {ent.mentions}×
                            </span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                  <div
                    role="button"
                    tabIndex={0}
                    onMouseDown={handleDragStart}
                    className="shrink-0 h-2 flex items-center justify-center cursor-ns-resize border-t border-[var(--border)] bg-[var(--bg-primary)] hover:bg-[var(--border)] transition-colors"
                    aria-label="Drag to resize"
                  >
                    <GripHorizontal size={12} className="text-[var(--text-secondary)]" />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Entity types breakdown */}
          {typeEntries.length > 0 && (
            <div className="mt-4">
              <h4 className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wide mb-2">
                Entity Types
              </h4>
              <div className="flex flex-wrap gap-2">
                {typeEntries.map(([type, count]) => (
                  <span
                    key={type}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-full bg-[var(--bg-primary)] border border-[var(--border)] text-[var(--text-primary)]"
                  >
                    <span className="w-2 h-2 rounded-full bg-[var(--accent)]" />
                    {type.replace(/_/g, " ")}
                    <span className="text-[var(--text-secondary)]">({count})</span>
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Top entities */}
          {stats && stats.top_entities.length > 0 && (
            <div className="mt-4">
              <h4 className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wide mb-2">
                Top Entities
              </h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
                {stats.top_entities.slice(0, 10).map((ent) => (
                  <div
                    key={ent.name}
                    className="flex items-center justify-between px-3 py-1.5 text-xs rounded-lg bg-[var(--bg-primary)]"
                  >
                    <span className="text-[var(--text-primary)] font-medium truncate mr-2">
                      {ent.name}
                    </span>
                    <div className="flex items-center gap-2 shrink-0">
                      {ent.types.length > 0 && (
                        <span className="text-[var(--text-secondary)]">
                          {ent.types[0].replace(/_/g, " ")}
                        </span>
                      )}
                      <span className="text-[var(--text-secondary)]">
                        {ent.mentions}x
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Classification model dropdown + Refresh button */}
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2">
              <label htmlFor="kb-model" className="text-xs font-medium text-[var(--text-secondary)] whitespace-nowrap">
                Classification model:
              </label>
              <select
                id="kb-model"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="text-sm bg-[var(--bg-primary)] border border-[var(--border)] rounded-lg px-3 py-2 text-[var(--text-primary)] min-w-[180px]"
              >
                <option value="">Default ({defaultModel})</option>
                {ollamaModels
                  .filter((m) => m !== defaultModel)
                  .map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
              </select>
            </div>
            <button
              onClick={handleRefresh}
              disabled={refreshing}
              className="flex items-center gap-2 px-4 py-2 text-sm bg-[var(--accent)] text-white rounded-lg hover:bg-[var(--accent-hover)] disabled:opacity-50 transition-colors"
            >
              <RefreshCw size={14} className={refreshing ? "animate-spin" : ""} />
              {refreshing ? "Rebuilding..." : "Refresh Knowledge Base"}
            </button>
            <span className="text-xs text-[var(--text-secondary)]">
              Rebuilds taxonomy from all document graphs
            </span>
          </div>
          {refreshMessage && (
            <div className="mt-2 text-sm text-[var(--text-secondary)] bg-[var(--bg-primary)] px-3 py-2 rounded-lg">
              {refreshMessage}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function StatCard({
  id,
  icon,
  label,
  value,
  isOpen,
  onClick,
}: {
  id: string;
  icon: React.ReactNode;
  label: string;
  value: number;
  isOpen: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      data-kb-stat-card
      onClick={onClick}
      className={`flex items-center gap-2.5 px-3 py-2.5 rounded-lg bg-[var(--bg-primary)] border transition-colors text-left w-full ${
        isOpen
          ? "border-[var(--accent)] ring-1 ring-[var(--accent)]/20"
          : "border-transparent hover:border-[var(--border)] hover:bg-[var(--bg-secondary)]"
      }`}
    >
      <div className="text-[var(--accent)]">{icon}</div>
      <div className="min-w-0">
        <div className="text-lg font-bold text-[var(--text-primary)] leading-tight">{value.toLocaleString()}</div>
        <div className="text-xs text-[var(--text-secondary)]">{label}</div>
      </div>
      <ChevronDown
        size={12}
        className={`ml-auto shrink-0 text-[var(--text-secondary)] transition-transform ${isOpen ? "rotate-180" : ""}`}
      />
    </button>
  );
}
