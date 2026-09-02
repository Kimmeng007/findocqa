"""A single shared throttle every Gemini call goes through — ours and
RAGAS's judge-LLM calls alike.

Week 3's first eval run failed because pacing was applied only to our own
generation calls; RAGAS's internal metric-scoring calls ran unthrottled
with their own concurrency, and the two together blew past the free
tier's 15-requests/minute cap. Worse, RAGAS's retry/backoff on a 429 adds
*more* calls to an already-saturated limit rather than easing off it. A
lock-protected minimum interval, applied at the actual point every call
departs to Gemini, closes that gap regardless of which code path (or how
many concurrent workers) is making the call.
"""

import threading
import time

# 15 requests/min is the hard cap; pace a bit under it for safety margin.
_MIN_INTERVAL_SECONDS = 4.5

_lock = threading.Lock()
_last_call_time = 0.0


def throttle() -> None:
    global _last_call_time
    with _lock:
        now = time.monotonic()
        elapsed = now - _last_call_time
        if elapsed < _MIN_INTERVAL_SECONDS:
            time.sleep(_MIN_INTERVAL_SECONDS - elapsed)
        _last_call_time = time.monotonic()
