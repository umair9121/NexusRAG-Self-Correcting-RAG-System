"""
Hardware detection module.

Detects CPU (Intel / AMD), GPU (NVIDIA / AMD / Intel Arc), and NPU presence
so the frontend can enable or disable vendor-specific performance options.
"""

from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from typing import Any


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class CPUInfo:
    vendor: str = "Unknown"          # "Intel", "AMD", "Unknown"
    model_name: str = ""
    core_count: int = 0
    has_avx2: bool = False
    has_avx512: bool = False
    has_amx: bool = False            # Intel AMX (Sapphire Rapids+)
    has_npu: bool = False            # Intel NPU / AMD XDNA presence


@dataclass
class GPUInfo:
    vendor: str = "Unknown"          # "NVIDIA", "AMD", "Intel"
    model_name: str = ""
    vram_mb: int = 0
    driver_version: str = ""
    cuda_version: str = ""           # NVIDIA only
    rocm_available: bool = False     # AMD only


@dataclass
class HardwareProfile:
    cpu: CPUInfo = field(default_factory=CPUInfo)
    gpus: list[GPUInfo] = field(default_factory=list)
    os: str = ""
    platform_arch: str = ""
    available_vendors: list[str] = field(default_factory=list)


# ── Detection helpers ────────────────────────────────────────────────────────

def _run(cmd: list[str], **kwargs: Any) -> str:
    """Run a command and return stripped stdout, or empty string on failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10, **kwargs)
        return r.stdout.strip()
    except Exception:
        return ""


def _detect_cpu() -> CPUInfo:
    info = CPUInfo()
    system = platform.system()

    if system == "Windows":
        # Use WMIC / PowerShell for model name
        raw = _run(["powershell", "-NoProfile", "-Command",
                     "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"])
        if raw:
            info.model_name = raw.strip()

        # Core count
        raw_cores = _run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Processor | Select-Object -First 1).NumberOfLogicalProcessors"])
        try:
            info.core_count = int(raw_cores.strip())
        except ValueError:
            import os as _os
            info.core_count = _os.cpu_count() or 0

        # CPU features via registry (limited, but helpful)
        raw_id = _run(["powershell", "-NoProfile", "-Command",
                       "(Get-CimInstance Win32_Processor | Select-Object -First 1).Description"])
        combined = (info.model_name + " " + raw_id).lower()

    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read()
        except OSError:
            cpuinfo = ""

        m = re.search(r"model name\s*:\s*(.+)", cpuinfo)
        if m:
            info.model_name = m.group(1).strip()

        cores = re.findall(r"^processor\s*:", cpuinfo, re.MULTILINE)
        import os as _os
        info.core_count = len(cores) if cores else (_os.cpu_count() or 0)

        flags_m = re.search(r"flags\s*:\s*(.+)", cpuinfo)
        flags = flags_m.group(1).lower() if flags_m else ""
        info.has_avx2 = "avx2" in flags
        info.has_avx512 = "avx512" in flags or "avx512f" in flags
        info.has_amx = "amx_tile" in flags or "amx_int8" in flags or "amx_bf16" in flags
        combined = info.model_name.lower()

    else:
        # macOS / other
        raw = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if raw:
            info.model_name = raw
        import os as _os
        info.core_count = _os.cpu_count() or 0
        combined = info.model_name.lower()

    # Vendor heuristic
    if "intel" in combined:
        info.vendor = "Intel"
    elif "amd" in combined:
        info.vendor = "AMD"

    # NPU heuristic: Intel Meteor Lake+ have NPUs, AMD has XDNA
    ml = info.model_name.lower()
    if "core ultra" in ml or "meteor lake" in ml or "lunar lake" in ml or "arrow lake" in ml:
        info.has_npu = True
    if "ryzen ai" in ml or "xdna" in ml:
        info.has_npu = True

    return info


def _detect_gpus() -> list[GPUInfo]:
    gpus: list[GPUInfo] = []

    # ── NVIDIA via nvidia-smi ────────────────────────────────────────────
    if shutil.which("nvidia-smi"):
        raw = _run([
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ])
        if raw:
            for line in raw.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    vram = 0
                    try:
                        vram = int(float(parts[1]))
                    except ValueError:
                        pass
                    gpu = GPUInfo(
                        vendor="NVIDIA",
                        model_name=parts[0],
                        vram_mb=vram,
                        driver_version=parts[2],
                    )
                    # Try to get CUDA version
                    cuda_raw = _run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
                    if cuda_raw:
                        gpu.cuda_version = cuda_raw.strip().splitlines()[0].strip()
                    gpus.append(gpu)

    # ── AMD via rocm-smi ────────────────────────────────────────────────
    if shutil.which("rocm-smi"):
        raw = _run(["rocm-smi", "--showproductname", "--json"])
        if raw:
            try:
                data = json.loads(raw)
                for _key, val in data.items():
                    if isinstance(val, dict):
                        name = val.get("Card Series", val.get("Card series", "AMD GPU"))
                        gpus.append(GPUInfo(
                            vendor="AMD",
                            model_name=str(name),
                            rocm_available=True,
                        ))
            except (json.JSONDecodeError, AttributeError):
                pass
    # Fallback: AMD on Windows via PowerShell
    if not any(g.vendor == "AMD" for g in gpus):
        raw = _run(["powershell", "-NoProfile", "-Command",
                     "Get-CimInstance Win32_VideoController | Where-Object {$_.Name -like '*AMD*' -or $_.Name -like '*Radeon*'} | Select-Object -ExpandProperty Name"])
        if raw:
            for line in raw.strip().splitlines():
                line = line.strip()
                if line:
                    gpus.append(GPUInfo(vendor="AMD", model_name=line))

    # ── Intel Arc / integrated via PowerShell / lspci ────────────────────
    system = platform.system()
    if system == "Windows":
        raw = _run(["powershell", "-NoProfile", "-Command",
                     "Get-CimInstance Win32_VideoController | Where-Object {$_.Name -like '*Intel*'} | Select-Object -ExpandProperty Name"])
        if raw:
            for line in raw.strip().splitlines():
                line = line.strip()
                if line:
                    gpus.append(GPUInfo(vendor="Intel", model_name=line))
    elif system == "Linux":
        raw = _run(["lspci"])
        if raw:
            for line in raw.splitlines():
                lower = line.lower()
                if "vga" in lower or "3d" in lower or "display" in lower:
                    if "intel" in lower:
                        gpus.append(GPUInfo(vendor="Intel", model_name=line.split(":")[-1].strip()))

    return gpus


# ── Public API ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def detect_hardware() -> HardwareProfile:
    """Detect hardware once and cache for the lifetime of the process."""
    cpu = _detect_cpu()
    gpus = _detect_gpus()

    vendors: set[str] = {"General"}
    if cpu.vendor == "Intel":
        vendors.add("Intel CPU")
    elif cpu.vendor == "AMD":
        vendors.add("AMD CPU")
    for g in gpus:
        if g.vendor == "NVIDIA":
            vendors.add("NVIDIA GPU")
        elif g.vendor == "AMD":
            vendors.add("AMD GPU")
        elif g.vendor == "Intel":
            vendors.add("Intel GPU")

    return HardwareProfile(
        cpu=cpu,
        gpus=gpus,
        os=platform.system(),
        platform_arch=platform.machine(),
        available_vendors=sorted(vendors),
    )


def hardware_profile_dict() -> dict[str, Any]:
    """Return the hardware profile as a plain dict (JSON-safe)."""
    profile = detect_hardware()
    d = asdict(profile)
    return d
