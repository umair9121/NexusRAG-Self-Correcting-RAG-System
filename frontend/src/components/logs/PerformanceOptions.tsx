"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Cpu,
  MonitorSpeaker,
  Zap,
  Shield,
  AlertTriangle,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  Info,
  Check,
  X,
} from "lucide-react";
import { getPerformanceOptions, updatePerformanceOption } from "@/lib/api";
import type { PerformanceOption, PerformanceOptionsResponse, HardwareProfile } from "@/lib/types";

// ── Category styling / icons ────────────────────────────────────────────

const CATEGORY_META: Record<
  string,
  { icon: typeof Cpu; color: string; accent: string; description: string }
> = {
  General: {
    icon: Zap,
    color: "text-emerald-400",
    accent: "border-emerald-500/40",
    description: "Optimizations available on all hardware",
  },
  "NVIDIA GPU": {
    icon: MonitorSpeaker,
    color: "text-green-400",
    accent: "border-green-500/40",
    description: "CUDA & Tensor Core accelerations for NVIDIA GPUs",
  },
  "AMD GPU": {
    icon: MonitorSpeaker,
    color: "text-red-400",
    accent: "border-red-500/40",
    description: "ROCm & HIP accelerations for AMD GPUs",
  },
  "Intel GPU": {
    icon: MonitorSpeaker,
    color: "text-blue-400",
    accent: "border-blue-500/40",
    description: "Xe & OpenVINO accelerations for Intel Arc / integrated GPUs",
  },
  "Intel CPU": {
    icon: Cpu,
    color: "text-blue-300",
    accent: "border-blue-400/40",
    description: "OpenVINO, IPEX, AMX, and NPU offload for Intel CPUs",
  },
  "AMD CPU": {
    icon: Cpu,
    color: "text-orange-400",
    accent: "border-orange-500/40",
    description: "ZenDNN & XDNA NPU offload for AMD CPUs",
  },
};

const CATEGORY_ORDER = [
  "General",
  "NVIDIA GPU",
  "AMD GPU",
  "Intel GPU",
  "Intel CPU",
  "AMD CPU",
];

// ── Component ───────────────────────────────────────────────────────────

export default function PerformanceOptions() {
  const [data, setData] = useState<PerformanceOptionsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [toggling, setToggling] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const [showHwInfo, setShowHwInfo] = useState(false);

  const fetchOptions = useCallback(async () => {
    try {
      setError(null);
      const resp = await getPerformanceOptions();
      setData(resp);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load performance options");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchOptions();
  }, [fetchOptions]);

  const handleToggle = async (key: string, currentEnabled: boolean) => {
    if (toggling) return;
    setToggling(key);
    try {
      const resp = await updatePerformanceOption(key, !currentEnabled);
      setData(resp);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update option");
    } finally {
      setToggling(null);
    }
  };

  const toggleCollapse = (cat: string) => {
    setCollapsed((prev) => ({ ...prev, [cat]: !prev[cat] }));
  };

  // ── Render states ─────────────────────────────────────────────────────

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full py-20">
        <RefreshCw size={20} className="animate-spin text-[var(--text-secondary)]" />
        <span className="ml-2 text-sm text-[var(--text-secondary)]">Detecting hardware...</span>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="flex flex-col items-center justify-center h-full py-20 gap-3">
        <AlertTriangle size={24} className="text-[var(--error)]" />
        <span className="text-sm text-[var(--error)]">{error}</span>
        <button
          onClick={() => { setLoading(true); fetchOptions(); }}
          className="text-xs text-[var(--accent)] hover:underline"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!data) return null;

  const hw = data.hardware;
  const cats = data.categories;

  return (
    <div className="flex flex-col h-full">
      {/* Header bar */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-[var(--border)] bg-[var(--bg-secondary)]">
        <Shield size={14} className="text-[var(--accent)]" />
        <span className="text-xs font-medium text-[var(--text-primary)]">Performance Boosts</span>
        <div className="flex-1" />

        {/* Hardware info toggle */}
        <button
          onClick={() => setShowHwInfo(!showHwInfo)}
          className="flex items-center gap-1 text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
        >
          <Info size={12} />
          Hardware Info
        </button>

        <button
          onClick={() => { setLoading(true); fetchOptions(); }}
          className="p-1 rounded text-[var(--text-secondary)] hover:text-[var(--accent)]"
          title="Re-detect hardware"
        >
          <RefreshCw size={14} />
        </button>
      </div>

      {/* Error banner */}
      {error && (
        <div className="px-4 py-2 bg-red-500/10 border-b border-red-500/30 text-xs text-red-400">
          {error}
        </div>
      )}

      {/* Hardware info panel (collapsible) */}
      {showHwInfo && <HardwareInfoPanel hw={hw} />}

      {/* Options list */}
      <div className="flex-1 overflow-y-auto">
        {CATEGORY_ORDER.map((catName) => {
          const options = cats[catName];
          if (!options || options.length === 0) return null;

          const meta = CATEGORY_META[catName] || CATEGORY_META.General;
          const Icon = meta.icon;
          const isCollapsed = collapsed[catName] ?? false;
          const isAvailable = options.some((o) => o.available);
          const enabledCount = options.filter((o) => o.enabled).length;

          return (
            <div key={catName} className={`border-b border-[var(--border)]`}>
              {/* Category header */}
              <button
                onClick={() => toggleCollapse(catName)}
                className="w-full flex items-center gap-3 px-4 py-3 hover:bg-[var(--bg-tertiary)] transition-colors"
              >
                {isCollapsed ? (
                  <ChevronRight size={14} className="text-[var(--text-secondary)]" />
                ) : (
                  <ChevronDown size={14} className="text-[var(--text-secondary)]" />
                )}
                <Icon size={16} className={meta.color} />
                <span className="text-sm font-medium text-[var(--text-primary)]">{catName}</span>

                {!isAvailable && catName !== "General" && (
                  <span className="ml-2 px-1.5 py-0.5 text-[10px] rounded bg-yellow-500/15 text-yellow-400 font-medium">
                    NOT DETECTED
                  </span>
                )}
                {isAvailable && enabledCount > 0 && (
                  <span className="ml-2 px-1.5 py-0.5 text-[10px] rounded bg-emerald-500/15 text-emerald-400 font-medium">
                    {enabledCount} ACTIVE
                  </span>
                )}

                <div className="flex-1" />
                <span className="text-[10px] text-[var(--text-secondary)]">
                  {options.length} option{options.length !== 1 ? "s" : ""}
                </span>
              </button>

              {/* Category description */}
              {!isCollapsed && (
                <div className="px-4 pb-1 pl-12">
                  <p className="text-[11px] text-[var(--text-secondary)]">{meta.description}</p>
                </div>
              )}

              {/* Options */}
              {!isCollapsed && (
                <div className="pb-2">
                  {options.map((opt) => (
                    <OptionRow
                      key={opt.key}
                      option={opt}
                      toggling={toggling === opt.key}
                      onToggle={() => handleToggle(opt.key, opt.enabled)}
                      accentBorder={meta.accent}
                    />
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Sub-components ──────────────────────────────────────────────────────

function OptionRow({
  option,
  toggling,
  onToggle,
  accentBorder,
}: {
  option: PerformanceOption;
  toggling: boolean;
  onToggle: () => void;
  accentBorder: string;
}) {
  const disabled = !option.available;

  return (
    <div
      className={`flex items-center gap-3 mx-4 my-1 px-3 py-2.5 rounded-lg border transition-colors ${
        disabled
          ? "border-[var(--border)] opacity-40 cursor-not-allowed"
          : option.enabled
          ? `${accentBorder} bg-white/[0.03]`
          : "border-transparent hover:bg-[var(--bg-tertiary)]"
      }`}
    >
      {/* Toggle switch */}
      <button
        disabled={disabled || toggling}
        onClick={onToggle}
        className={`relative shrink-0 w-9 h-5 rounded-full transition-colors duration-200 ${
          disabled
            ? "bg-[var(--bg-tertiary)]"
            : option.enabled
            ? "bg-[var(--accent)]"
            : "bg-[var(--bg-tertiary)] border border-[var(--border)]"
        }`}
        aria-label={`Toggle ${option.label}`}
      >
        <span
          className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform duration-200 flex items-center justify-center ${
            option.enabled ? "translate-x-4" : "translate-x-0"
          }`}
        >
          {toggling ? (
            <RefreshCw size={8} className="animate-spin text-gray-500" />
          ) : option.enabled ? (
            <Check size={8} className="text-[var(--accent)]" />
          ) : (
            <X size={8} className="text-gray-400" />
          )}
        </span>
      </button>

      {/* Label & description */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className={`text-sm font-medium ${disabled ? "text-[var(--text-secondary)]" : "text-[var(--text-primary)]"}`}>
            {option.label}
          </span>
          {disabled && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--bg-tertiary)] text-[var(--text-secondary)]">
              Unavailable
            </span>
          )}
        </div>
        <p className="text-[11px] text-[var(--text-secondary)] mt-0.5 leading-tight truncate">
          {option.description}
        </p>
      </div>
    </div>
  );
}

function HardwareInfoPanel({ hw }: { hw: HardwareProfile }) {
  return (
    <div className="px-4 py-3 bg-[var(--bg-tertiary)] border-b border-[var(--border)] space-y-2">
      <div className="text-xs font-medium text-[var(--text-primary)] mb-2">Detected Hardware</div>

      {/* CPU */}
      <div className="flex items-start gap-2">
        <Cpu size={12} className="text-blue-400 mt-0.5 shrink-0" />
        <div className="text-[11px]">
          <span className="text-[var(--text-primary)] font-medium">CPU:</span>{" "}
          <span className="text-[var(--text-secondary)]">
            {hw.cpu.model_name || "Unknown"} ({hw.cpu.vendor}, {hw.cpu.core_count} threads)
          </span>
          {(hw.cpu.has_avx2 || hw.cpu.has_avx512 || hw.cpu.has_amx || hw.cpu.has_npu) && (
            <div className="flex gap-1 mt-1 flex-wrap">
              {hw.cpu.has_avx2 && <Badge label="AVX2" />}
              {hw.cpu.has_avx512 && <Badge label="AVX-512" />}
              {hw.cpu.has_amx && <Badge label="AMX" />}
              {hw.cpu.has_npu && <Badge label="NPU" />}
            </div>
          )}
        </div>
      </div>

      {/* GPUs */}
      {hw.gpus.length > 0 ? (
        hw.gpus.map((gpu, i) => (
          <div key={i} className="flex items-start gap-2">
            <MonitorSpeaker size={12} className={
              gpu.vendor === "NVIDIA" ? "text-green-400 mt-0.5 shrink-0" :
              gpu.vendor === "AMD" ? "text-red-400 mt-0.5 shrink-0" :
              "text-blue-400 mt-0.5 shrink-0"
            } />
            <div className="text-[11px]">
              <span className="text-[var(--text-primary)] font-medium">GPU {i + 1}:</span>{" "}
              <span className="text-[var(--text-secondary)]">
                {gpu.model_name || "Unknown"} ({gpu.vendor})
                {gpu.vram_mb > 0 && ` — ${(gpu.vram_mb / 1024).toFixed(1)} GB VRAM`}
              </span>
              {(gpu.driver_version || gpu.cuda_version || gpu.rocm_available) && (
                <div className="flex gap-1 mt-1 flex-wrap">
                  {gpu.driver_version && <Badge label={`Driver ${gpu.driver_version}`} />}
                  {gpu.cuda_version && <Badge label={`CUDA ${gpu.cuda_version}`} />}
                  {gpu.rocm_available && <Badge label="ROCm" />}
                </div>
              )}
            </div>
          </div>
        ))
      ) : (
        <div className="flex items-center gap-2">
          <MonitorSpeaker size={12} className="text-[var(--text-secondary)] shrink-0" />
          <span className="text-[11px] text-[var(--text-secondary)]">No discrete GPU detected</span>
        </div>
      )}

      {/* Platform */}
      <div className="text-[10px] text-[var(--text-secondary)] pt-1 border-t border-[var(--border)]">
        {hw.os} · {hw.platform_arch} · Detected vendors: {hw.available_vendors.join(", ")}
      </div>
    </div>
  );
}

function Badge({ label }: { label: string }) {
  return (
    <span className="px-1.5 py-0.5 text-[10px] rounded bg-[var(--bg-secondary)] text-[var(--text-secondary)] border border-[var(--border)]">
      {label}
    </span>
  );
}
