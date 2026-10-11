"""System resource probing for the router's resource filter.

RAM and CPU come from psutil. GPU memory comes from NVML (the
``nvidia-ml-py`` package), falling back to ``nvidia-smi``. VRAM held by
models the backend has loaded is reported separately, because the backend
can evict those models to make room for another one.
"""
import logging
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, Optional

import psutil

from .backends import InferenceBackend, normalize_tag

logger = logging.getLogger(__name__)

GIB = 1024 ** 3


class GpuProbe:
    """Total and free VRAM of GPU 0, or None when no NVIDIA GPU is found."""

    def __init__(self):
        self._nvml = None
        self._handle = None
        self.source = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml = pynvml
            self.source = "nvml"
        except Exception:
            if shutil.which("nvidia-smi"):
                self.source = "nvidia-smi"

    def read(self) -> Optional[Dict[str, Any]]:
        try:
            if self._nvml is not None:
                mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                name = self._nvml.nvmlDeviceGetName(self._handle)
                return {"name": name if isinstance(name, str) else name.decode(),
                        "total_vram_gb": mem.total / GIB, "free_vram_gb": mem.free / GIB,
                        "used_vram_gb": mem.used / GIB, "source": "nvml"}
            if self.source == "nvidia-smi":
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
                name, total, free, used = [p.strip() for p in out.split(",")]
                return {"name": name, "total_vram_gb": float(total) / 1024,
                        "free_vram_gb": float(free) / 1024, "used_vram_gb": float(used) / 1024,
                        "source": "nvidia-smi"}
        except Exception as e:
            logger.debug(f"GPU probe failed: {e}")
        return None


class ResourceMonitor:
    """Cached snapshot of RAM, CPU, VRAM and backend-loaded models.

    A probe takes about 10 ms on Windows (psutil.virtual_memory alone is
    ~8 ms), so with ``background=True`` a daemon thread refreshes the
    snapshot every ``ttl`` seconds and requests never wait on a probe.
    Otherwise the snapshot is refreshed lazily when older than ``ttl``.
    """

    def __init__(self, backend: Optional[InferenceBackend] = None, ttl: float = 2.0,
                 gpu_probe: Optional[GpuProbe] = None, background: bool = False):
        self.backend = backend
        self.ttl = ttl
        self.gpu_probe = gpu_probe if gpu_probe is not None else GpuProbe()
        self._snapshot: Optional[Dict[str, Any]] = None
        self._taken_at = 0.0
        self._started_at = float("-inf")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.background = background
        if background:
            self.snapshot(refresh=True)
            threading.Thread(target=self._refresh_loop, name="resource-monitor", daemon=True).start()

    def _refresh_loop(self):
        while not self._stop.wait(self.ttl):
            try:
                self.snapshot(refresh=True)
            except Exception as e:
                logger.debug(f"Resource refresh failed: {e}")

    def stop(self):
        self._stop.set()

    def snapshot(self, refresh: bool = False) -> Dict[str, Any]:
        stale = time.monotonic() - self._taken_at > self.ttl
        if refresh or self._snapshot is None or (stale and not self.background):
            started = time.monotonic()
            snap = self._probe()
            with self._lock:
                # A probe that started earlier (e.g. the background thread,
                # mid-load) must not overwrite a newer one
                if started >= self._started_at:
                    self._snapshot = snap
                    self._started_at = started
                    self._taken_at = time.monotonic()
        return self._snapshot

    def _probe(self) -> Dict[str, Any]:
        vm = psutil.virtual_memory()
        snap: Dict[str, Any] = {
            "available_ram_gb": vm.available / GIB,
            "total_ram_gb": vm.total / GIB,
            "cpu_usage_percent": psutil.cpu_percent(),
            "available_cpu_cores": psutil.cpu_count(),
            "disk_usage_percent": psutil.disk_usage("/").percent,
            "gpu": self.gpu_probe.read(),
            "backend_loaded_vram_gb": {},
        }
        if self.backend is not None:
            try:
                for tag, info in self.backend.loaded_models().items():
                    vram = info.get("size_vram_bytes")
                    if vram is not None:
                        snap["backend_loaded_vram_gb"][normalize_tag(tag)] = vram / GIB
            except Exception as e:
                logger.debug(f"Could not read backend loaded models: {e}")
        return snap


def model_fits(model, resources: Dict[str, Any]) -> bool:
    """Whether ``model`` (a ModelInfo or ModelConfig) fits in current resources.

    With a GPU, an LLM needs its measured VRAM footprint to fit in free VRAM
    plus the VRAM the backend's loaded models hold (the backend can evict
    them). A model that is already loaded always fits. Without a GPU, or for
    models with no VRAM requirement, the RAM requirement applies instead.
    """
    req = model.resource_requirements
    if resources.get("available_cpu_cores", 1) < req.get("cpu_cores", 1):
        return False

    loaded = resources.get("backend_loaded_vram_gb", {})
    tag = getattr(model, "ollama_tag", None) or getattr(model, "backend_model", None)
    if tag and normalize_tag(tag) in loaded:
        return True

    need_vram = req.get("min_vram_gb", 0)
    gpu = resources.get("gpu")
    if need_vram and gpu:
        budget = gpu["free_vram_gb"] + sum(loaded.values())
        return need_vram <= budget

    return resources.get("available_ram_gb", 0) >= req.get("min_ram_gb", 0)
