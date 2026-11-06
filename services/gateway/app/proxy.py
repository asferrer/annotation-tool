"""
Shared HTTP proxy helper for forwarding requests to the segmentation service.
"""

import os
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

SEGMENTATION_SERVICE_URL = os.environ.get(
    "SEGMENTATION_SERVICE_URL", "http://segmentation:8002"
)


async def proxy_segmentation(
    method: str,
    path: str,
    data: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 180.0,
) -> Any:
    """Forward a request to the segmentation service and return its JSON body."""
    url = f"{SEGMENTATION_SERVICE_URL}{path}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if method == "GET":
                resp = await client.get(url, params=params)
            elif method == "POST":
                resp = await client.post(url, json=data)
            elif method == "DELETE":
                resp = await client.delete(url, params=params)
            else:
                raise HTTPException(405, "Method not allowed")

            if resp.status_code >= 400:
                raise HTTPException(resp.status_code, resp.text)
            return resp.json()
    except httpx.RequestError as e:
        raise HTTPException(503, f"Segmentation service unavailable: {e}")
