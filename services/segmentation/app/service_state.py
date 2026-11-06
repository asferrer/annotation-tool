"""
Shared service state and initialization for the segmentation service.

All routers import state, helper functions, and shared resources from here.
"""

import os
import sys
import time
import asyncio
import logging
import base64
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from app.scene_analyzer import SemanticSceneAnalyzer
from app.object_extractor import ObjectExtractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared utilities imports (with fallback)
# ---------------------------------------------------------------------------
try:
    from shared.vram_monitor import VRAMMonitor
    VRAM_MONITOR_AVAILABLE = True
except ImportError:
    try:
        sys.path.insert(0, '/app')
        from shared.vram_monitor import VRAMMonitor
        VRAM_MONITOR_AVAILABLE = True
    except ImportError:
        VRAM_MONITOR_AVAILABLE = False
        VRAMMonitor = None

try:
    from shared.job_database import JobDatabase, get_job_db
    JOB_DATABASE_AVAILABLE = True
except ImportError:
    try:
        sys.path.insert(0, '/app')
        from shared.job_database import JobDatabase, get_job_db
        JOB_DATABASE_AVAILABLE = True
    except ImportError:
        JOB_DATABASE_AVAILABLE = False
        JobDatabase = None
        get_job_db = None


# ---------------------------------------------------------------------------
# Service state
# ---------------------------------------------------------------------------
class ServiceState:
    """Global service state"""
    scene_analyzer = None
    debug_scene_analyzer = None
    sam3_model = None
    sam3_processor = None
    sam3_tracker_model = None      # Sam3TrackerModel for interactive point-based segmentation
    sam3_tracker_processor = None  # Sam3TrackerProcessor
    device: str = "cpu"
    sam3_available: bool = False
    sam3_loading: bool = False
    sam3_load_error: Optional[str] = None
    sam3_load_progress: str = ""
    sam3_model_id: Optional[str] = None
    gpu_available: bool = False
    object_extractor: Optional["ObjectExtractor"] = None
    db: Optional["JobDatabase"] = None
    _loading_task: Optional[asyncio.Task] = None


state = ServiceState()

# Job tracking for async operations
extraction_jobs: Dict[str, Dict[str, Any]] = {}
sam3_conversion_jobs: Dict[str, Dict[str, Any]] = {}
labeling_jobs: Dict[str, Dict[str, Any]] = {}

# Thread pool for CPU-bound operations
thread_pool = ThreadPoolExecutor(max_workers=2)

# Concurrency control for labeling jobs
MAX_CONCURRENT_LABELING_JOBS = 1
MAX_CONCURRENT_IMAGES_PER_JOB = 4
labeling_job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LABELING_JOBS)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def encode_region_map_base64(region_map: np.ndarray) -> Optional[str]:
    """Encode region map as base64 PNG string."""
    success, encoded = cv2.imencode('.png', region_map)
    if success:
        return base64.b64encode(encoded.tobytes()).decode('utf-8')
    return None


def _calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate IoU between two COCO-format bounding boxes [x, y, w, h]."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - intersection_area

    if union_area == 0:
        return 0.0
    return intersection_area / union_area


# ---------------------------------------------------------------------------
# Initialization functions
# ---------------------------------------------------------------------------
def init_scene_analyzer():
    """Initialize scene analyzers with SAM3 model if available."""
    state.scene_analyzer = SemanticSceneAnalyzer(
        use_sam3=state.sam3_available,
        device="cuda" if state.gpu_available else "cpu",
        sam3_model=state.sam3_model,
        sam3_processor=state.sam3_processor,
    )

    debug_output_dir = "/shared/segmentation/debug"
    os.makedirs(debug_output_dir, exist_ok=True)
    state.debug_scene_analyzer = SemanticSceneAnalyzer(
        use_sam3=state.sam3_available,
        device="cuda" if state.gpu_available else "cpu",
        sam3_model=state.sam3_model,
        sam3_processor=state.sam3_processor,
        debug=True,
        debug_output_dir=debug_output_dir,
    )
    logger.info(f"Scene analyzers initialized (SAM3: {state.sam3_available})")


def init_object_extractor():
    """Initialize object extractor with shared SAM3 model."""
    state.object_extractor = ObjectExtractor(
        sam3_model=state.sam3_model,
        sam3_processor=state.sam3_processor,
        device=state.device,
    )
    logger.info(f"ObjectExtractor initialized (SAM3: {state.sam3_available})")


def init_gpu():
    """Initialize GPU detection."""
    import torch
    state.gpu_available = torch.cuda.is_available()
    state.device = "cuda" if state.gpu_available else "cpu"

    if state.gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU available: {gpu_name} ({gpu_mem:.1f}GB VRAM)")
    else:
        logger.warning("No GPU available, using CPU")


def _load_sam3_sync():
    """Load SAM3 model synchronously (called from background thread)."""
    import torch

    state.sam3_load_progress = "Starting SAM3 load..."

    try:
        from transformers import Sam3Processor, Sam3Model

        model_id = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")
        hf_token = os.environ.get("HF_TOKEN")

        if not hf_token:
            logger.warning("[Background] HF_TOKEN not set — facebook/sam3 is a gated model and requires a valid token")

        logger.info(f"[Background] Loading SAM3 model: {model_id}")
        state.sam3_load_progress = f"Loading processor from {model_id}..."
        load_start = time.time()

        try:
            state.sam3_processor = Sam3Processor.from_pretrained(model_id, token=hf_token)
        except Exception as proc_err:
            cause = proc_err.__cause__ or proc_err.__context__
            root = f" (caused by: {cause})" if cause else ""
            raise RuntimeError(
                f"Failed to load Sam3Processor for '{model_id}'{root}. "
                f"If the model is gated, accept the terms at https://huggingface.co/{model_id} "
                f"and ensure HF_TOKEN is set and has 'read' scope."
            ) from proc_err
        proc_time = time.time() - load_start
        logger.info(f"[Background] Processor loaded in {proc_time:.1f}s")
        state.sam3_load_progress = "Loading model weights..."

        model_start = time.time()
        use_fp16 = state.gpu_available and os.environ.get("SAM3_FP32", "").lower() != "true"

        if use_fp16:
            logger.info("[Background] Loading model in FP16 (half precision)...")
            state.sam3_load_progress = "Loading model in FP16 mode..."
            state.sam3_model = Sam3Model.from_pretrained(
                model_id, token=hf_token, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            ).to(state.device)
        else:
            logger.info("[Background] Loading model in FP32 (full precision)...")
            state.sam3_load_progress = "Loading model in FP32 mode..."
            state.sam3_model = Sam3Model.from_pretrained(
                model_id, token=hf_token, low_cpu_mem_usage=True,
            ).to(state.device)

        state.sam3_model.eval()
        model_time = time.time() - model_start
        logger.info(f"[Background] Model loaded in {model_time:.1f}s")

        # Sam3TrackerModel is NOT loaded: facebook/sam3 checkpoint lacks the interactive
        # decoder weights (mask_decoder.*, prompt_encoder.*) so it would produce garbage.
        # Instead, Sam3TrackerProcessor is loaded for its input_points/input_labels API,
        # and combined with Sam3Model (which provides the decoder) + input_ids from
        # Sam3Processor for point-guided segmentation.
        state.sam3_tracker_model = None
        try:
            from transformers import Sam3TrackerProcessor
            tracker_proc_start = time.time()
            state.sam3_tracker_processor = Sam3TrackerProcessor.from_pretrained(model_id, token=hf_token)
            logger.info(f"[Background] Sam3TrackerProcessor loaded in {time.time() - tracker_proc_start:.1f}s")
        except Exception as tp_err:
            logger.warning(f"[Background] Sam3TrackerProcessor not available: {tp_err}")
            state.sam3_tracker_processor = None

        state.sam3_load_progress = "Initializing scene analyzer..."
        init_scene_analyzer()
        init_object_extractor()

        state.sam3_available = True
        state.sam3_model_id = model_id
        state.sam3_load_progress = "Ready"
        total_time = time.time() - load_start
        logger.info(f"[Background] SAM3 fully initialized in {total_time:.1f}s total")

    except ImportError as e:
        error_msg = f"SAM3 not available (transformers may need update): {e}"
        logger.warning(f"[Background] {error_msg}")
        state.sam3_load_error = error_msg
        state.sam3_available = False
        init_scene_analyzer()
        init_object_extractor()
    except Exception as e:
        error_msg = f"SAM3 loading failed: {e}"
        logger.warning(f"[Background] {error_msg}")
        state.sam3_load_error = error_msg
        state.sam3_available = False
        init_scene_analyzer()
        init_object_extractor()
    finally:
        state.sam3_loading = False


async def init_sam3_background():
    """Initialize SAM3 model in background (non-blocking)."""
    state.sam3_loading = True
    state.sam3_load_error = None
    state.sam3_load_progress = "Queued for loading..."
    logger.info("Starting SAM3 background loading...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(thread_pool, _load_sam3_sync)


def _reacquire_sam3_via_manager() -> None:
    """Swap SAM3 back into VRAM after another engine (video/insid3) used the GPU."""
    from app.model_manager import manager, ModelBusyError  # late import (avoids cycle)
    try:
        manager.acquire("sam3")
        manager.release("sam3")
    except ModelBusyError as e:
        logger.warning(f"Cannot reload SAM3: {e}")
    except Exception as e:
        logger.warning(f"SAM3 reload via ModelManager failed: {e}")


async def wait_for_sam3(timeout: float = 300.0) -> bool:
    """Wait for SAM3 to finish loading, reloading it if it was swapped out."""
    if state.sam3_available:
        return True
    if not state.sam3_loading:
        if state.sam3_load_error:
            return False
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(thread_pool, _reacquire_sam3_via_manager)
        return state.sam3_available
    start = time.time()
    while state.sam3_loading and (time.time() - start) < timeout:
        await asyncio.sleep(0.5)
    return state.sam3_available
