"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { FileText, MessageSquare, Activity, Search, SlidersHorizontal, Cpu, HardDrive, FolderKanban } from "lucide-react";
import { useEffect, useState } from "react";
import { getSystemHealth } from "@/lib/api";
import type { SystemHealth } from "@/lib/types";

const NAV_ITEMS = [
  { href: "/", label: "Documents", icon: FileText },
  { href: "/chat", label: "Chat", icon: MessageSquare },
  { href: "/workspaces", label: "Workspaces", icon: FolderKanban },
  { href: "/explore", label: "Explore", icon: Search },
  { href: "/logs", label: "Logs", icon: Activity },
  { href: "/options", label: "Options", icon: SlidersHorizontal },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [logoError, setLogoError] = useState(false);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const data = await getSystemHealth();
        setHealth(data);
      } catch {
        setHealth(null);
      }
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <aside className="w-64 h-screen flex flex-col border-r border-[var(--border)] bg-[var(--bg-secondary)]">
      {/* Header */}
      <div className="p-4 border-b border-[var(--border)]">
        {!logoError ? (
          <div className="relative w-full h-12">
            <Image
              src="/images/Logo.png"
              alt=" RAG System & Generation for EDI Guidelines"
              fill
              className="object-contain object-left"
              onError={() => setLogoError(true)}
              unoptimized
            />
          </div>
        ) : null}
        <h1 className={`text-lg font-bold text-[var(--text-primary)] ${!logoError ? "mt-2" : ""}`}>
          RAG System & Generation for EDI Guidelines
        </h1>
        <p className="text-xs text-[var(--text-secondary)] mt-0.5">Adaptive Retrieval System</p>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-3 space-y-1">
        {NAV_ITEMS.map((item) => {
          const active = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                active
                  ? "bg-[var(--accent)] text-white"
                  : "text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)] hover:text-[var(--text-primary)]"
              }`}
            >
              <item.icon size={18} />
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* System Status */}
      <div className="p-4 border-t border-[var(--border)] space-y-3">
        <div className="flex items-center gap-2 text-xs">
          <Cpu size={14} />
          <span className="text-[var(--text-secondary)]">Ollama:</span>
          <span className={health?.ollama_connected ? "text-[var(--success)]" : "text-[var(--error)]"}>
            {health?.ollama_connected ? "Online" : "Offline"}
          </span>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <HardDrive size={14} />
          <span className="text-[var(--text-secondary)]">RAM:</span>
          <span className="text-[var(--text-primary)]">{health?.available_ram_gb ?? "??"} GB</span>
        </div>
        <div className="text-xs text-[var(--text-secondary)]">
          {health?.total_documents ?? 0} docs ({health?.documents_ready ?? 0} ready)
        </div>
      </div>
    </aside>
  );
}
