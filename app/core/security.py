"""Lightweight protection for a publicly deployed backend.

- Shared-secret header (X-API-Key): only your Streamlit frontend knows it.
- Global sliding-window rate limit on expensive endpoints: caps Gemini spend even if the key leaks.
Both are no-ops when not configured, so local development stays frictionless.
"""
import hmac
import threading
import time
from collections import deque
from typing import Optional

from fastapi import Header, HTTPException

from app.core.config import settings

_calls: deque = deque()
_lock = threading.Lock()


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = settings.app_api_key
    if not expected:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def rate_limit() -> None:
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return
    now = time.time()
    with _lock:
        while _calls and now - _calls[0] > 60:
            _calls.popleft()
        if len(_calls) >= limit:
            retry_after = int(60 - (now - _calls[0])) + 1
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit reached ({limit}/min). Retry in {retry_after}s.",
                headers={"Retry-After": str(retry_after)},
            )
        _calls.append(now)


def reset_rate_limiter() -> None:
    """Test helper."""
    with _lock:
        _calls.clear()
