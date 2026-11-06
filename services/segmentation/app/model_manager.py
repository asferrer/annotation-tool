"""
ModelManager: single-GPU model swapping between SAM3 (image), SAM3 Video
and future engines (INSID3).

Only one model family is resident in VRAM at a time. acquire() swaps models
on demand: it refuses (ModelBusyError) while another model has active users,
unloads the current one, and loads the requested one. Designed to be called
from worker threads (loads can take minutes and must not block the event loop).
"""

import gc
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from app.service_state import (
    state,
    labeling_jobs,
    extraction_jobs,
    sam3_conversion_jobs,
    _load_sam3_sync,
)

logger = logging.getLogger(__name__)

_ACTIVE_JOB_STATUSES = {"pending", "queued", "running", "processing"}


class ModelBusyError(RuntimeError):
    def __init__(self, active_model: str, reason: str):
        super().__init__(f"Model '{active_model}' is busy: {reason}")
        self.active_model = active_model
        self.reason = reason


def _legacy_sam3_busy() -> Optional[str]:
    """Return a description if any legacy (non-manager) SAM3 job is active."""
    for name, jobs in (
        ("labeling", labeling_jobs),
        ("extraction", extraction_jobs),
        ("sam3_conversion", sam3_conversion_jobs),
    ):
        for job_id, job in jobs.items():
            status = job.get("status", "")
            status = status.value if hasattr(status, "value") else str(status)
            if status in _ACTIVE_JOB_STATUSES:
                return f"{name} job {job_id} is {status}"
    return None


@dataclass
class _ManagedModel:
    name: str
    loader: Callable[[], Any]
    unloader: Callable[[Any], None]
    can_unload: Callable[[], Optional[str]] = lambda: None  # returns busy reason
    handle: Any = None
    refcount: int = 0


class ModelManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._models: Dict[str, _ManagedModel] = {}
        self.active: Optional[str] = None
        self.swapping: bool = False

    def register(
        self,
        name: str,
        loader: Callable[[], Any],
        unloader: Callable[[Any], None],
        can_unload: Callable[[], Optional[str]] = lambda: None,
    ) -> None:
        self._models[name] = _ManagedModel(name, loader, unloader, can_unload)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            self._reconcile()
            return {
                "active_model": self.active,
                "swapping": self.swapping,
                "registered": sorted(self._models),
                "refcounts": {n: m.refcount for n, m in self._models.items()},
            }

    def _reconcile(self) -> None:
        """SAM3 may have been loaded at startup outside the manager — adopt it."""
        if self.active is None and state.sam3_available and "sam3" in self._models:
            self._models["sam3"].handle = {
                "model": state.sam3_model, "processor": state.sam3_processor,
            }
            self.active = "sam3"

    def acquire(self, name: str) -> Any:
        """Load (swapping if needed) and pin the requested model. Returns its handle."""
        if name not in self._models:
            raise KeyError(f"Unknown model: {name}")
        with self._lock:
            self._reconcile()
            target = self._models[name]
            if self.active == name and target.handle is not None:
                target.refcount += 1
                return target.handle

            if self.active is not None and self.active != name:
                current = self._models[self.active]
                busy = self._busy_reason(current)
                if busy:
                    raise ModelBusyError(self.active, busy)
                self._unload_locked(current)

            self.swapping = True
            try:
                target.handle = self._load_with_retry(target)
                self.active = name
                target.refcount += 1
                return target.handle
            finally:
                self.swapping = False

    def release(self, name: str) -> None:
        with self._lock:
            model = self._models.get(name)
            if model and model.refcount > 0:
                model.refcount -= 1

    def _busy_reason(self, model: _ManagedModel) -> Optional[str]:
        if model.refcount > 0:
            return f"{model.refcount} active job(s)"
        return model.can_unload()

    def _unload_locked(self, model: _ManagedModel) -> None:
        logger.info(f"[ModelManager] Unloading '{model.name}' to free VRAM")
        try:
            model.unloader(model.handle)
        finally:
            model.handle = None
            self.active = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    free, total = torch.cuda.mem_get_info()
                    logger.info(
                        f"[ModelManager] VRAM after unload: {free / 1e9:.1f}GB free / {total / 1e9:.1f}GB"
                    )
            except Exception:
                pass

    def _load_with_retry(self, model: _ManagedModel) -> Any:
        logger.info(f"[ModelManager] Loading '{model.name}'")
        start = time.time()
        try:
            handle = model.loader()
        except Exception as first_err:
            logger.warning(
                f"[ModelManager] First load of '{model.name}' failed ({first_err}); "
                "retrying after cache cleanup"
            )
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            handle = model.loader()
        logger.info(f"[ModelManager] '{model.name}' loaded in {time.time() - start:.1f}s")
        return handle


# ---------------------------------------------------------------------------
# Loaders / unloaders
# ---------------------------------------------------------------------------

def _sam3_loader() -> Any:
    if not state.sam3_available:
        state.sam3_loading = True
        try:
            _load_sam3_sync()
        finally:
            state.sam3_loading = False
        if not state.sam3_available:
            raise RuntimeError(state.sam3_load_error or "SAM3 failed to load")
    return {"model": state.sam3_model, "processor": state.sam3_processor}


def _sam3_unloader(_handle: Any) -> None:
    state.sam3_available = False
    state.sam3_model = None
    state.sam3_processor = None
    state.sam3_tracker_processor = None
    state.scene_analyzer = None
    state.debug_scene_analyzer = None
    state.object_extractor = None


def _sam3_video_loader() -> Any:
    import torch
    from transformers import Sam3VideoModel, Sam3VideoProcessor

    model_id = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")
    hf_token = os.environ.get("HF_TOKEN")
    use_fp16 = state.gpu_available and os.environ.get("SAM3_FP32", "").lower() != "true"
    dtype = torch.float16 if use_fp16 else torch.float32

    processor = Sam3VideoProcessor.from_pretrained(model_id, token=hf_token)
    model = Sam3VideoModel.from_pretrained(
        model_id, token=hf_token, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(state.device)
    model.eval()
    return {"model": model, "processor": processor, "dtype": dtype}


def _sam3_video_unloader(handle: Any) -> None:
    if isinstance(handle, dict):
        handle.pop("model", None)
        handle.pop("processor", None)


def _insid3_loader() -> Any:
    from app.insid3.loader import load_insid3

    return load_insid3(state.device)


def _insid3_unloader(_handle: Any) -> None:
    pass  # gc + empty_cache in _unload_locked frees the frozen encoder


manager = ModelManager()
manager.register("sam3", _sam3_loader, _sam3_unloader, can_unload=_legacy_sam3_busy)
manager.register("sam3_video", _sam3_video_loader, _sam3_video_unloader)
manager.register("insid3", _insid3_loader, _insid3_unloader)
