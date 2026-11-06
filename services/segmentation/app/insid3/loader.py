"""
INSID3 loader: frozen DINOv3 encoder (local gated weights) + vendored model.

The DINOv3 weights are gated by Meta and must be downloaded manually into
DINOV3_WEIGHTS_DIR (default /app/checkpoints/dinov3). The torch.hub repo code
for facebookresearch/dinov3 is cached under TORCH_HOME (checkpoints volume).
"""

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(os.environ.get("DINOV3_WEIGHTS_DIR", "/app/checkpoints/dinov3"))
MODEL_SIZE = os.environ.get("INSID3_MODEL_SIZE", "large")

_HUB_NAMES = {
    "small": "dinov3_vits16",
    "base": "dinov3_vitb16",
    "large": "dinov3_vitl16",
}
_WEIGHT_FILES = {
    "small": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "base": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "large": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
}


def weights_status() -> dict:
    """Report whether the DINOv3 weights are present (for /models/status and UI)."""
    expected = WEIGHTS_DIR / _WEIGHT_FILES.get(MODEL_SIZE, _WEIGHT_FILES["large"])
    return {
        "model_size": MODEL_SIZE,
        "weights_path": str(expected),
        "weights_present": expected.exists(),
        "instructions": (
            None if expected.exists() else
            f"Download {_WEIGHT_FILES[MODEL_SIZE]} from Meta "
            f"(https://ai.meta.com/resources/models-and-libraries/dinov3/, gated: "
            f"requires accepting the DINOv3 License) and place it at {expected}"
        ),
    }


def load_insid3(device: str) -> Any:
    """Build the INSID3 model. Raises RuntimeError with actionable message."""
    import torch

    status = weights_status()
    if not status["weights_present"]:
        raise RuntimeError(status["instructions"])

    # Cache the torch.hub dinov3 repo in the checkpoints volume
    os.environ.setdefault("TORCH_HOME", "/app/checkpoints/torch")

    logger.info(f"Loading DINOv3 {MODEL_SIZE} encoder from {status['weights_path']}")
    encoder = torch.hub.load(
        "facebookresearch/dinov3",
        _HUB_NAMES[MODEL_SIZE],
        weights=status["weights_path"],
    )

    from app.insid3.vendor import INSID3

    model = INSID3(encoder=encoder, device=device).to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    logger.info("INSID3 ready (training-free, frozen DINOv3)")
    return model
