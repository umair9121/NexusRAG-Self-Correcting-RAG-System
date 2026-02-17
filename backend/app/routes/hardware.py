"""
Hardware detection and performance-options endpoints.

* GET  /api/hardware/profile          → detected CPU / GPU / NPU info
* GET  /api/hardware/options          → all perf options grouped by category
* PATCH /api/hardware/options         → toggle a single option on/off
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException

from app.config import STORAGE_DIR
from app.modules.hardware import detect_hardware, hardware_profile_dict
from app.schemas import (
    HardwareProfileResponse,
    PerformanceOption,
    PerformanceOptionsResponse,
    PerformanceOptionUpdate,
)

router = APIRouter(prefix="/api/hardware", tags=["hardware"])

# Persist user-toggled options to a JSON file so they survive restarts
OPTIONS_FILE = STORAGE_DIR / "performance_options.json"


# ── Option definitions ───────────────────────────────────────────────────
# Each tuple: (key, label, description, category, default_enabled)

_OPTION_DEFS: list[tuple[str, str, str, str, bool]] = [
    # ── General (all hardware) ────────────────────────────────────────────
    ("thread_pinning", "CPU Thread Pinning",
     "Pin inference threads to physical cores to reduce context-switching overhead",
     "General", False),
    ("memory_pool", "Pre-allocated Memory Pool",
     "Reserve a contiguous memory pool at startup for faster tensor allocation",
     "General", False),
    ("quantization_auto", "Auto Quantization",
     "Automatically quantize models to the optimal format for your hardware",
     "General", False),
    ("flash_attention", "Flash Attention",
     "Use memory-efficient Flash Attention for faster transformer inference",
     "General", False),
    ("batch_scheduling", "Dynamic Batch Scheduling",
     "Batch concurrent requests together to improve GPU utilization",
     "General", False),
    ("mmap_loading", "Memory-Mapped Model Loading",
     "Load model weights via mmap for faster startup and reduced RAM usage",
     "General", True),

    # ── NVIDIA GPU ────────────────────────────────────────────────────────
    ("cuda_graphs", "CUDA Graphs",
     "Capture and replay CUDA graph for reduced kernel-launch overhead",
     "NVIDIA GPU", False),
    ("tensor_cores", "Tensor Core Acceleration",
     "Prefer Tensor Core (FP16/BF16/INT8) paths for matrix operations",
     "NVIDIA GPU", False),
    ("cudnn_benchmark", "cuDNN Auto-Tuner",
     "Enable cuDNN benchmark mode to auto-select the fastest convolution algorithm",
     "NVIDIA GPU", False),
    ("nvidia_mps", "Multi-Process Service (MPS)",
     "Allow multiple processes to share the GPU via NVIDIA MPS for better utilization",
     "NVIDIA GPU", False),
    ("tensorrt_optimize", "TensorRT Optimization",
     "Use TensorRT to optimize and accelerate model inference on NVIDIA GPUs",
     "NVIDIA GPU", False),

    # ── AMD GPU ───────────────────────────────────────────────────────────
    ("rocm_hip", "ROCm HIP Acceleration",
     "Use ROCm HIP runtime for GPU-accelerated inference on AMD GPUs",
     "AMD GPU", False),
    ("rocm_flash_attn", "ROCm Flash Attention",
     "Enable composable-kernel-based Flash Attention for AMD RDNA/CDNA GPUs",
     "AMD GPU", False),
    ("rocm_tuning", "rocBLAS Tuning",
     "Run rocBLAS auto-tuning to find optimal GEMM kernels for your AMD GPU",
     "AMD GPU", False),
    ("amd_gpu_mem_pool", "HIP Memory Pool",
     "Use HIP memory pooling to reduce allocation overhead on AMD GPUs",
     "AMD GPU", False),

    # ── Intel GPU (Arc / integrated) ──────────────────────────────────────
    ("xmx_accel", "XMX Acceleration",
     "Enable Intel Xe Matrix Extensions for accelerated matrix ops on Arc GPUs",
     "Intel GPU", False),
    ("intel_gpu_openvino", "OpenVINO GPU Backend",
     "Run inference on Intel GPU via the OpenVINO toolkit",
     "Intel GPU", False),
    ("level_zero", "Level-Zero Runtime",
     "Use oneAPI Level-Zero low-level runtime for minimal-overhead GPU dispatch",
     "Intel GPU", False),

    # ── Intel CPU ─────────────────────────────────────────────────────────
    ("openvino_cpu", "OpenVINO CPU Backend",
     "Use the OpenVINO toolkit for optimized inference on Intel CPUs",
     "Intel CPU", False),
    ("ipex_optimize", "IPEX Optimization",
     "Apply Intel Extension for PyTorch (IPEX) optimizations to the inference path",
     "Intel CPU", False),
    ("amx_accel", "AMX Acceleration",
     "Enable Intel Advanced Matrix Extensions (AMX) for BF16/INT8 matmul on Sapphire Rapids+",
     "Intel CPU", False),
    ("npu_offload", "NPU Offload",
     "Offload supported operators to the Intel NPU (Meteor Lake+ / Arrow Lake+)",
     "Intel CPU", False),
    ("avx512_opt", "AVX-512 Optimization",
     "Use AVX-512 VNNI / BF16 instruction paths for accelerated vector ops",
     "Intel CPU", False),
    ("onednn_verbose", "oneDNN Verbose Logging",
     "Enable verbose oneDNN logging for profiling Intel-optimized primitives",
     "Intel CPU", False),

    # ── AMD CPU ───────────────────────────────────────────────────────────
    ("amd_zen_opt", "Zen Microarch Tuning",
     "Enable Zen-specific memory and scheduler optimizations (ZenDNN / AOCL)",
     "AMD CPU", False),
    ("amd_smu_boost", "Precision Boost Override",
     "Request the SMU to maintain higher boost clocks during inference workloads",
     "AMD CPU", False),
    ("amd_npu_offload", "AMD NPU Offload (XDNA)",
     "Offload supported operators to the AMD XDNA NPU (Ryzen AI series)",
     "AMD CPU", False),
    ("amd_prefetch", "Software Prefetch Tuning",
     "Use Zen-optimized software prefetch distances for large tensor traversals",
     "AMD CPU", False),
]


# ── Helpers ──────────────────────────────────────────────────────────────

def _load_saved_options() -> dict[str, bool]:
    """Load user-persisted toggle states from disk."""
    if OPTIONS_FILE.exists():
        try:
            return json.loads(OPTIONS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_options(data: dict[str, bool]) -> None:
    OPTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    OPTIONS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _build_options() -> dict[str, list[PerformanceOption]]:
    """Build the full option set, merging saved toggles and hardware availability."""
    hw = detect_hardware()
    saved = _load_saved_options()

    categories: dict[str, list[PerformanceOption]] = {}

    for key, label, desc, category, default_on in _OPTION_DEFS:
        # Determine availability based on detected hardware
        available = True
        if category == "NVIDIA GPU":
            available = any(g.vendor == "NVIDIA" for g in hw.gpus)
        elif category == "AMD GPU":
            available = any(g.vendor == "AMD" for g in hw.gpus)
        elif category == "Intel GPU":
            available = any(g.vendor == "Intel" for g in hw.gpus)
        elif category == "Intel CPU":
            available = hw.cpu.vendor == "Intel"
        elif category == "AMD CPU":
            available = hw.cpu.vendor == "AMD"
        # "General" is always available

        # Special sub-checks
        if key == "npu_offload":
            available = available and hw.cpu.has_npu
        if key == "amx_accel":
            available = available and hw.cpu.has_amx
        if key == "avx512_opt":
            available = available and hw.cpu.has_avx512
        if key == "amd_npu_offload":
            available = available and hw.cpu.has_npu

        enabled = saved.get(key, default_on) if available else False

        opt = PerformanceOption(
            key=key,
            label=label,
            description=desc,
            enabled=enabled,
            available=available,
            category=category,
        )
        categories.setdefault(category, []).append(opt)

    return categories


# ── Endpoints ────────────────────────────────────────────────────────────

@router.get("/profile", response_model=HardwareProfileResponse)
async def get_hardware_profile():
    """Return detected CPU/GPU information."""
    return hardware_profile_dict()


@router.get("/options", response_model=PerformanceOptionsResponse)
async def get_performance_options():
    """Return all performance options grouped by category with availability flags."""
    hw = hardware_profile_dict()
    cats = _build_options()
    return PerformanceOptionsResponse(
        categories=cats,
        hardware=HardwareProfileResponse(**hw),
    )


@router.patch("/options", response_model=PerformanceOptionsResponse)
async def update_performance_option(body: PerformanceOptionUpdate):
    """Toggle a single performance option on or off."""
    # Validate the key exists
    valid_keys = {d[0] for d in _OPTION_DEFS}
    if body.key not in valid_keys:
        raise HTTPException(status_code=404, detail=f"Unknown option key: {body.key}")

    # Check availability
    hw = detect_hardware()
    opt_def = next(d for d in _OPTION_DEFS if d[0] == body.key)
    category = opt_def[3]

    def _is_available() -> bool:
        if category == "NVIDIA GPU":
            return any(g.vendor == "NVIDIA" for g in hw.gpus)
        if category == "AMD GPU":
            return any(g.vendor == "AMD" for g in hw.gpus)
        if category == "Intel GPU":
            return any(g.vendor == "Intel" for g in hw.gpus)
        if category == "Intel CPU":
            return hw.cpu.vendor == "Intel"
        if category == "AMD CPU":
            return hw.cpu.vendor == "AMD"
        return True  # General

    if not _is_available():
        raise HTTPException(
            status_code=403,
            detail=f"Option '{body.key}' requires {category} hardware which was not detected.",
        )

    # Persist
    saved = _load_saved_options()
    saved[body.key] = body.enabled
    _save_options(saved)

    # Return full refreshed state
    cats = _build_options()
    hw_dict = hardware_profile_dict()
    return PerformanceOptionsResponse(
        categories=cats,
        hardware=HardwareProfileResponse(**hw_dict),
    )
