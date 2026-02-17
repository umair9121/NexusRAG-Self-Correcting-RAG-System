"use client";

import { Sun, Moon } from "lucide-react";
import PerformanceOptions from "@/components/logs/PerformanceOptions";
import { useTheme } from "@/lib/theme";

export default function OptionsPage() {
  const { theme, setTheme } = useTheme();

  return (
    <div className="h-full flex flex-col">
      <div className="px-6 py-4 border-b border-[var(--border)]">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold">Options</h1>
            <p className="text-sm text-[var(--text-secondary)] mt-1">
              Performance boosts and hardware-specific settings for NVIDIA, AMD, and Intel
            </p>
          </div>
          <div className="flex items-center gap-1 rounded-lg p-1 bg-[var(--bg-tertiary)] border border-[var(--border)]">
            <button
              type="button"
              onClick={() => setTheme("light")}
              className={`flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                theme === "light"
                  ? "bg-[var(--accent)] text-white"
                  : "text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-secondary)]"
              }`}
              aria-pressed={theme === "light"}
              aria-label="Light mode"
            >
              <Sun size={18} />
              Light
            </button>
            <button
              type="button"
              onClick={() => setTheme("dark")}
              className={`flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                theme === "dark"
                  ? "bg-[var(--accent)] text-white"
                  : "text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-secondary)]"
              }`}
              aria-pressed={theme === "dark"}
              aria-label="Dark mode"
            >
              <Moon size={18} />
              Dark
            </button>
          </div>
        </div>
      </div>
      <div className="flex-1 overflow-hidden">
        <PerformanceOptions />
      </div>
    </div>
  );
}
